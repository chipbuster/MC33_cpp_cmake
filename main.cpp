#include "source/MC33.cpp"
//#include "source/grid3d.cpp"
//#include "source/surface.cpp"

//#include "include/MC33.h"

#include <iostream>
#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <igl/file_dialog_open.h>
#include <igl/file_dialog_save.h>
#include <igl/AABB.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_edge_normals.h>
#include <igl/pseudonormal_test.h>
#include <igl/grid.h>
#include <igl/copyleft/quadprog.h>
#include <tbb/tbb.h>

#include "polyscope/polyscope.h"
#include "polyscope/messages.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"

Eigen::MatrixXd refV, MC33V;
Eigen::MatrixXi refF, MC33F;
igl::AABB<Eigen::MatrixXd, 3> tree;
double offset = 0.01; 
double bboxsize = 1;

double implicitSurf(double x, double y, double z)
{
	double lower_bound = std::numeric_limits<double>::min();
	double upper_bound = std::numeric_limits<double>::max();
	const double max_abs = std::max(std::abs(lower_bound), std::abs(upper_bound));
	const double up_sqr_d = std::pow(max_abs, 2.0);
	const double low_sqr_d =
		std::pow(std::max(max_abs - (upper_bound - lower_bound), (double)0.0), 2.0);
	
	Eigen::Matrix<double, 1, 3> c, p;
	p << x, y, z;
	int i = -1;
	double sqrd = tree.squared_distance(refV, refF, p, low_sqr_d, up_sqr_d, i, c);

	Eigen::Matrix<double, 1, 3> n3;
	double s = 1;

	if (sqrd >= up_sqr_d || sqrd < low_sqr_d)
		return std::numeric_limits<double>::quiet_NaN();
	else
		return s * std::sqrt(sqrd);
}

void callback()
{
	ImGui::PushItemWidth(100);
	if (ImGui::CollapsingHeader("I/O", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::Button("Load", ImVec2(-1, 0))) {
			std::string inputPath = igl::file_dialog_open();
			std::replace(inputPath.begin(), inputPath.end(), '\\', '/'); // handle the backslash issue for windows

			if (!igl::read_triangle_mesh(inputPath, refV, refF))
			{
				std::cerr << "missing the mesh file!" << std::endl;
				exit(EXIT_FAILURE);
			}
			polyscope::registerSurfaceMesh("reference mesh", refV, refF);

			tree.init(refV, refF);

			Eigen::Vector3d m = refV.colwise().minCoeff();
			Eigen::Vector3d M = refV.colwise().maxCoeff();

			bboxsize = (M - m).norm();
		}
		if (ImGui::Button("save", ImVec2(-1, 0))) {
			std::string outputPath = igl::file_dialog_save();
			std::replace(outputPath.begin(), outputPath.end(), '\\', '/'); // handle the backslash issue for windows
			igl::write_triangle_mesh(outputPath, MC33V, MC33F);
		}
	}

	if (ImGui::CollapsingHeader("offset Options", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::InputDouble("distance", &offset))
		{
			if (offset < 0)
				offset = 0.01;
		}
	}

	if (ImGui::Button("compute MC33", ImVec2(-1, 0)))
	{
		Eigen::Vector3d m = refV.colwise().minCoeff();
		Eigen::Vector3d M = refV.colwise().maxCoeff();

		double bboxsize = (M - m).norm();
		double testDist = 0.1 * bboxsize;	// a slightly larger bbox

		Eigen::RowVector3d minCorner, maxCorner;
		for (int i = 0; i < 3; i++)
		{
			minCorner(i) = m(i) - 2 * testDist;
			maxCorner(i) = M(i) + 2 * testDist;
		}

		int k = std::ceil(std::log2(1 / offset));
		int resx = std::pow(2, k);
		int resy = std::pow(2, k);
		int resz = std::pow(2, k);

		grid3d G;
		G.generate_grid_from_fn(
			minCorner[0], minCorner[1], minCorner[2],
			maxCorner[0], maxCorner[1], maxCorner[2],
			(maxCorner[0] - minCorner[0]) / resx, (maxCorner[1] - minCorner[1]) / resy, (maxCorner[2] - minCorner[2]) / resz, implicitSurf);

		MC33 MC;
		MC.set_grid3d(G);

		surface S;
		MC.calculate_isosurface(S, offset * bboxsize);

		int nverts = S.get_num_vertices();
		int nfaces = S.get_num_triangles();
		MC33V.resize(nverts, 3);
		MC33F.resize(nfaces, 3);

		tbb::parallel_for(
			tbb::blocked_range<int>(0u, nverts, 10),
			[&](const tbb::blocked_range<int>& range)
			{
				for (int i = range.begin(); i != range.end(); ++i)
				{
					MC33V.row(i) << S.getVertex(i)[0], S.getVertex(i)[1], S.getVertex(i)[2];
				}
			}
		);

		tbb::parallel_for(
			tbb::blocked_range<int>(0u, nfaces, 10),
			[&](const tbb::blocked_range<int>& range)
			{
				for (int i = range.begin(); i != range.end(); ++i)
				{
					MC33F.row(i) << S.getTriangle(i)[0], S.getTriangle(i)[1], S.getTriangle(i)[2];
				}
			}
		);

		polyscope::registerSurfaceMesh("MC33 mesh", MC33V, MC33F);
		
	}

	ImGui::PopItemWidth();
}

int main(int argc, char** argv)
{
	std::string inputPath = "";
	if (argc < 2)
	{
		inputPath = igl::file_dialog_open();
	}
	else
		inputPath = argv[1];
	std::replace(inputPath.begin(), inputPath.end(), '\\', '/'); // handle the backslash issue for windows
	if (!igl::read_triangle_mesh(inputPath, refV, refF))
	{
		std::cout << "failed to load file." << std::endl;
		exit(EXIT_FAILURE);
	}


	// Options
	polyscope::options::autocenterStructures = true;
	polyscope::view::windowWidth = 1024;
	polyscope::view::windowHeight = 1024;

	// Initialize polyscope
	polyscope::init();

	polyscope::view::upDir = polyscope::view::UpDir::ZUp;

	// Add the callback
	polyscope::state::userCallback = callback;

	polyscope::options::groundPlaneHeightFactor = 0.25; // adjust the plane height

	polyscope::registerSurfaceMesh("reference mesh", refV, refF);

	// Show the gui
	polyscope::show();


	return 0;
}