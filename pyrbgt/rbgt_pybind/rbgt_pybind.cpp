// #include "src/evaluator.h"
#include "src/body.h"
#include "src/normal_image_viewer.h"
#include "src/occlusion_mask_renderer.h"
#include "src/region_modality.h"
#include "src/renderer_geometry.h"
#include "src/camera.h"
#include "src/model.h"
#include "src/renderer.h"
#include "src/common.h"

#include "ndarray_converter.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <experimental/filesystem>
#include <Eigen/Geometry>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

namespace py = pybind11;

using Transform3fA = Eigen::Transform<float, 3, Eigen::Affine>;

cv::Mat nparray_to_mat_uint8(py::array_t<uint8_t>& img)
{ 
    //auto im = img.unchecked<3>();
    auto rows = img.shape(0);
    auto cols = img.shape(1);
    auto type = CV_8UC3;

    cv::Mat img1(rows, cols, type, (unsigned char*)img.data());
    return img1;
}

py::dtype determine_np_dtype(int depth)
{
    switch (depth) {
    case CV_8U: return py::dtype::of<uint8_t>();
    case CV_8S: return py::dtype::of<int8_t>();
    case CV_16U: return py::dtype::of<uint16_t>();
    case CV_16S: return py::dtype::of<int16_t>();
    case CV_32S: return py::dtype::of<int32_t>();
    case CV_32F: return py::dtype::of<float>();
    case CV_64F: return py::dtype::of<double>();
    default:
        throw std::invalid_argument("Unsupported data type.");
    }
}

std::vector<std::size_t> determine_shape(cv::Mat& m)
{
    if (m.channels() == 1) {
        return {
            static_cast<size_t>(m.rows)
            , static_cast<size_t>(m.cols)
        };
    }

    return {
        static_cast<size_t>(m.rows)
        , static_cast<size_t>(m.cols)
        , static_cast<size_t>(m.channels())
    };
}

py::capsule make_capsule(cv::Mat& m)
{
    return py::capsule(new cv::Mat(m)
        , [](void *v) { delete reinterpret_cast<cv::Mat*>(v); }
        );
}

py::array mat_to_nparray(cv::Mat& m)
{
    if (!m.isContinuous()) {
        throw std::invalid_argument("Only continuous Mats supported.");
    }

    return py::array(determine_np_dtype(m.depth())
        , determine_shape(m)
        , m.data
        , make_capsule(m));
}

py::array mat_to_nparray_clone(cv::Mat& m_beforclone)
{
    cv::Mat m(m_beforclone.size(),m_beforclone.type());
    m_beforclone.copyTo(m);
    if (!m.isContinuous()) {
        throw std::invalid_argument("Only continuous Mats supported.");
    }

    return py::array(determine_np_dtype(m.depth())
        , determine_shape(m)
        , m.data
        , make_capsule(m));
}

std::experimental::filesystem::path filesystem(std::string &path) {
    std::experimental::filesystem::path fs = path;
    return fs;
}

std::string fs2string(std::experimental::filesystem::path &fs) {
    std::string path = fs;
    return path;
}

bool mat2t3fA(std::vector<float> &mat, Transform3fA &pose) {
    int ind = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            pose.matrix()(i, j) = mat[ind];
            ind = ind + 1;
        }
    }

    pose.matrix()(0, 3) = mat[9];
    pose.matrix()(1, 3) = mat[10];
    pose.matrix()(2, 3) = mat[11];

    return true;
}

bool mat2t3fAfullMat(std::vector<float> &mat, Transform3fA &pose) {
    pose.matrix() << mat[0], mat[1], mat[2],
      mat[3], mat[4], mat[5], mat[6], mat[7], mat[8],
      mat[9], mat[10], mat[11], mat[12], mat[13], mat[14], mat[15];
    return true;
}

std::vector<float> t3fA2mat(Transform3fA &pose) {
    std::vector<float> mat;
    mat.resize(16);
    int ind = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            mat[ind] = pose.matrix()(i, j);
            ind = ind + 1;
        }
    }
    return mat;
}

std::vector<float> vector3f2list(Eigen::Vector3f &vec) {
    std::vector<float> mat(vec.data(), vec.data() + vec.size());
    return mat;
}

std::vector<float> vector3b2list(cv::Vec3b &vec) {
    std::vector<float> mat;
    mat.resize(3);
    mat[0] = vec.val[0];
    mat[1] = vec.val[1];
    mat[2] = vec.val[2];
    return mat;
}

rbgt::Model::TemplateView getClosestTemplateView(const Transform3fA &body2camera_pose, rbgt::Model &model) {
    const rbgt::Model::TemplateView *template_view;
    model.GetClosestTemplateView(body2camera_pose, &template_view);
    return *template_view;
}

cv::Point2i list2cv2i(std::vector<int> &center) {
    cv::Point2i cv2i;
    cv2i.x = int(center[0]);
    cv2i.y = int(center[1]);
    return cv2i;
}

PYBIND11_MODULE(rbgt_pybind, m) {
    NDArrayConverter::init_numpy();
    // RBGT modules

    // -------------------------------------
    // Body
    py::class_<rbgt::Body, std::shared_ptr<rbgt::Body>> body(m, "Body");
    body.def(py::init<std::string&, std::experimental::filesystem::path&, float, bool, bool, float>());
    body.def(py::init<std::string&, std::experimental::filesystem::path&, float, bool, bool, float, const Transform3fA&>());
    // Geometry setters
    body.def("set_name", &rbgt::Body::set_name);
    body.def("set_geometry_path", &rbgt::Body::set_geometry_path);
    body.def("set_geometry_unit_in_meter", &rbgt::Body::set_geometry_unit_in_meter);
    body.def("set_geometry_counterclockwise", &rbgt::Body::set_geometry_counterclockwise);
    body.def("set_geometry_enable_culling", &rbgt::Body::set_geometry_enable_culling);
    body.def("set_maximum_body_diameter", &rbgt::Body::set_maximum_body_diameter);
    body.def("set_geometry2body_pose", &rbgt::Body::set_geometry2body_pose);
    body.def("set_occlusion_mask_id", &rbgt::Body::set_occlusion_mask_id);
    // Pose setters
    body.def("set_body2world_pose", &rbgt::Body::set_body2world_pose);
    body.def("set_world2body_pose", &rbgt::Body::set_world2body_pose);
    // Geometry getters
    body.def("name", &rbgt::Body::name);
    body.def("geometry_path", &rbgt::Body::geometry_path);
    body.def("geometry_unit_in_meter", &rbgt::Body::geometry_unit_in_meter);
    body.def("geometry_counterclockwise", &rbgt::Body::geometry_counterclockwise);
    body.def("geometry_enable_culling", &rbgt::Body::geometry_enable_culling);
    body.def("maximum_body_diameter", &rbgt::Body::maximum_body_diameter);
    body.def("geometry2body_pose", &rbgt::Body::geometry2body_pose);
    body.def("occlusion_mask_id", &rbgt::Body::occlusion_mask_id);
    // Pose getters
    body.def("body2world_pose", &rbgt::Body::body2world_pose);
    body.def("world2body_pose", &rbgt::Body::world2body_pose);
    body.def("geometry2world_pose", &rbgt::Body::geometry2world_pose);
    body.def("world2geometry_pose", &rbgt::Body::world2geometry_pose);

    // -------------------------------------
    // Viewer -> NOT DONE! 
    py::class_<rbgt::Viewer, std::shared_ptr<rbgt::Viewer>> viewer(m, "Viewer");
    // Setter
    viewer.def("set_display_images", &rbgt::Viewer::set_display_images);
    // Main
    viewer.def("StartSavingImages", &rbgt::Viewer::StartSavingImages);
    viewer.def("StopSavingImages", &rbgt::Viewer::StopSavingImages);
    // Getters
    viewer.def("name", &rbgt::Viewer::name);
    viewer.def("camera_ptr", &rbgt::Viewer::camera_ptr);
    viewer.def("save_path", &rbgt::Viewer::save_path);
    viewer.def("display_images", &rbgt::Viewer::display_images);
    viewer.def("save_images", &rbgt::Viewer::save_images);
    viewer.def("initialized", &rbgt::Viewer::initialized);

    // -------------------------------------
    // NormalImageViewer 
    py::class_<rbgt::NormalImageViewer, rbgt::Viewer, std::shared_ptr<rbgt::NormalImageViewer>> niv(m, "NormalImageViewer");
    niv.def(py::init<>());
    niv.def("Init", &rbgt::NormalImageViewer::Init);
    niv.def("set_opacity", &rbgt::NormalImageViewer::set_opacity);
    niv.def("UpdateViewer", &rbgt::NormalImageViewer::UpdateViewer);
    niv.def("normal_image", &rbgt::NormalImageViewer::normal_image);

    // -------------------------------------
    // Renderer -> NOT DONE! 
    py::class_<rbgt::Renderer, std::shared_ptr<rbgt::Renderer>> renderer(m, "Renderer");
    renderer.def("set_camera2world_pose", &rbgt::Renderer::set_camera2world_pose);
    renderer.def("InitFromCamera", &rbgt::Renderer::InitFromCamera);
    renderer.def("intrinsics", &rbgt::Renderer::intrinsics);
    renderer.def("name", &rbgt::Renderer::name);

    // -------------------------------------
    // Camera -> NOT DONE! 
    py::class_<rbgt::Camera, std::shared_ptr<rbgt::Camera>> cam(m, "Camera");
    cam.def(py::init<>());
    cam.def("set_camera2world_pose", &rbgt::Camera::set_camera2world_pose);
    cam.def("InitFromCamera", &rbgt::Camera::set_world2camera_pose);
    cam.def("intrinsics", &rbgt::Camera::set_save_index);
    cam.def("set_save_image_type", &rbgt::Camera::set_save_image_type);
    cam.def("set_save_index", &rbgt::Camera::set_save_index);
    cam.def("set_name", &rbgt::Camera::set_name);
    cam.def("StartSavingImages", &rbgt::Camera::StartSavingImages);
    cam.def("StopSavingImages", &rbgt::Camera::StopSavingImages);
    cam.def("UpdateImage2", &rbgt::Camera::UpdateImage2);
    cam.def("set_intrinsics", &rbgt::Camera::set_intrinsics);

    cam.def("image", &rbgt::Camera::image);
    cam.def("name", &rbgt::Camera::name);
    cam.def("intrinsics", &rbgt::Camera::intrinsics);
    cam.def("camera2world_pose", &rbgt::Camera::camera2world_pose);
    cam.def("world2camera_pose", &rbgt::Camera::world2camera_pose);
    cam.def("save_path", &rbgt::Camera::save_path);
    cam.def("save_index", &rbgt::Camera::save_index);
    cam.def("save_image_type", &rbgt::Camera::save_image_type);
    cam.def("save_images", &rbgt::Camera::save_images);

    // -------------------------------------
    // OcclusionMaskRenderer -> NOT DONE! 
    py::class_<rbgt::OcclusionMaskRenderer, rbgt::Renderer, std::shared_ptr<rbgt::OcclusionMaskRenderer>> omr(m, "OcclusionMaskRenderer");
    omr.def(py::init<>());
    omr.def("Init", &rbgt::OcclusionMaskRenderer::Init); // <- constructor/readwrite attribute for Transform3fA needed
    omr.def("InitFromCamera", &rbgt::OcclusionMaskRenderer::InitFromCamera);
    omr.def("set_intrinsics", &rbgt::OcclusionMaskRenderer::set_intrinsics);
    omr.def("set_z_min", &rbgt::OcclusionMaskRenderer::set_z_min);
    omr.def("set_mask_resolution", &rbgt::OcclusionMaskRenderer::set_mask_resolution);
    omr.def("set_dilation_radius", &rbgt::OcclusionMaskRenderer::set_dilation_radius);
    //  Main method
    omr.def("StartRendering", &rbgt::OcclusionMaskRenderer::StartRendering);
    omr.def("FetchOcclusionMask", &rbgt::OcclusionMaskRenderer::FetchOcclusionMask);
    // Getters for mask and internal variables
    omr.def("occlusion_mask", &rbgt::OcclusionMaskRenderer::occlusion_mask); // <- output trype cv::Mat to be wrapped
    omr.def("mask_resolution", &rbgt::OcclusionMaskRenderer::mask_resolution);
    omr.def("dilation_radius", &rbgt::OcclusionMaskRenderer::dilation_radius);

    // -------------------------------------
    // RegionModality -> NOT DONE! 
    py::class_<rbgt::RegionModality, std::shared_ptr<rbgt::RegionModality>> rm(m, "RegionModality");
    rm.def(py::init<>());
    rm.def("Init", &rbgt::RegionModality::Init); // <- constructor for model and camera needed
    // Debug Helpers
    rm.def_readwrite("function_lookup_f", &rbgt::RegionModality::function_lookup_f_);
    rm.def_readwrite("function_lookup_b", &rbgt::RegionModality::function_lookup_b_);
    rm.def("PrecalculateFunctionLookup", &rbgt::RegionModality::PrecalculateFunctionLookup);
    rm.def_readwrite("line_length_in_segments", &rbgt::RegionModality::line_length_in_segments_);
    rm.def_readwrite("distribution_length_minus_1_half", &rbgt::RegionModality::distribution_length_minus_1_half_);
    rm.def_readwrite("distribution_length_plus_1_half", &rbgt::RegionModality::distribution_length_plus_1_half_);
    rm.def_readwrite("max_abs_dloglikelihood_ddelta_cs", &rbgt::RegionModality::max_abs_dloglikelihood_ddelta_cs_);
    rm.def("PrecalculateDistributionVariables", &rbgt::RegionModality::PrecalculateDistributionVariables);
    rm.def("PrecalculateBodyVariables", &rbgt::RegionModality::PrecalculateBodyVariables);
    rm.def_readwrite("encoded_occlusion_mask_id", &rbgt::RegionModality::encoded_occlusion_mask_id_);
    rm.def("PrecalculateCameraVariables", &rbgt::RegionModality::PrecalculateCameraVariables);
    rm.def_readwrite("image_width_minus_1", &rbgt::RegionModality::image_width_minus_1_);
    rm.def("PrecalculatePoseVariables", &rbgt::RegionModality::PrecalculatePoseVariables);
    rm.def_readwrite("body2camera_pose", &rbgt::RegionModality::body2camera_pose_);
    rm.def("AddLinePixelColorsToTempHistograms", &rbgt::RegionModality::AddLinePixelColorsToTempHistograms);
    rm.def_readwrite("temp_histogram_f", &rbgt::RegionModality::temp_histogram_f_);
    rm.def_readwrite("temp_histogram_b", &rbgt::RegionModality::temp_histogram_b_);
    rm.def_readwrite("histogram_bitshift", &rbgt::RegionModality::histogram_bitshift_);
    rm.def_readwrite("histogram_f", &rbgt::RegionModality::histogram_f_);
    rm.def_readwrite("histogram_b", &rbgt::RegionModality::histogram_b_);

    // Setters for general distribution
    rm.def("set_n_points", &rbgt::RegionModality::set_n_points);
    rm.def("set_function_slope", &rbgt::RegionModality::set_function_slope);
    rm.def("set_function_length", &rbgt::RegionModality::set_function_length);
    rm.def("set_distribution_length", &rbgt::RegionModality::set_distribution_length);
    rm.def("set_scales", &rbgt::RegionModality::set_scales);
    rm.def("set_probability_threshold", &rbgt::RegionModality::set_probability_threshold);
    rm.def("set_min_continuous_distance", &rbgt::RegionModality::set_min_continuous_distance);
    rm.def("set_use_linear_function", &rbgt::RegionModality::set_use_linear_function);
    rm.def("set_use_const_variance", &rbgt::RegionModality::set_use_const_variance);
    // Setters for histogram calculation
    rm.def("set_n_histogram_bins", &rbgt::RegionModality::set_n_histogram_bins);
    rm.def("set_learning_rate_f", &rbgt::RegionModality::set_learning_rate_f);
    rm.def("set_learning_rate_b", &rbgt::RegionModality::set_learning_rate_b);
    rm.def("set_unconsidered_line_length", &rbgt::RegionModality::set_unconsidered_line_length);
    rm.def("set_considered_line_length", &rbgt::RegionModality::set_considered_line_length);
    // Setters for optimization
    rm.def("set_tikhonov_parameter_rotation", &rbgt::RegionModality::set_tikhonov_parameter_rotation);
    rm.def("set_tikhonov_parameter_translation", &rbgt::RegionModality::set_tikhonov_parameter_translation);
    // Setters for occlusion handling
    rm.def("UseOcclusionHandling", &rbgt::RegionModality::UseOcclusionHandling);
    rm.def("DoNotUseOcclusionHandling", &rbgt::RegionModality::DoNotUseOcclusionHandling);
    // Setters for general visualization settings
    rm.def("set_display_visualization", &rbgt::RegionModality::set_display_visualization);
    rm.def("StartSavingVisualizations", &rbgt::RegionModality::StartSavingVisualizations);
    rm.def("StopSavingVisualizations", &rbgt::RegionModality::StopSavingVisualizations);
    // Setters to turn on individual visualizations
    rm.def("set_visualize_lines_correspondence", &rbgt::RegionModality::set_visualize_lines_correspondence);
    rm.def("set_visualize_points_occlusion_mask_correspondence", &rbgt::RegionModality::set_visualize_points_occlusion_mask_correspondence);
    rm.def("set_visualize_points_pose_update", &rbgt::RegionModality::set_visualize_points_pose_update);
    rm.def("set_visualize_points_histogram_image_pose_update", &rbgt::RegionModality::set_visualize_points_histogram_image_pose_update);
    rm.def("set_visualize_points_result", &rbgt::RegionModality::set_visualize_points_result);
    rm.def("set_visualize_points_histogram_image_result", &rbgt::RegionModality::set_visualize_points_histogram_image_result);
    // Main methods
    rm.def("StartModality", &rbgt::RegionModality::StartModality);
    rm.def("CalculateBeforeCameraUpdate", &rbgt::RegionModality::CalculateBeforeCameraUpdate);
    rm.def("CalculateCorrespondences", &rbgt::RegionModality::CalculateCorrespondences);
    rm.def("VisualizeCorrespondences", &rbgt::RegionModality::VisualizeCorrespondences);
    rm.def("CalculatePoseUpdate", &rbgt::RegionModality::CalculatePoseUpdate);
    rm.def("VisualizePoseUpdate", &rbgt::RegionModality::VisualizePoseUpdate);
    rm.def("VisualizeResults", &rbgt::RegionModality::VisualizeResults);
    // Getters data
    rm.def("name", &rbgt::RegionModality::name);
    rm.def("body_ptr", &rbgt::RegionModality::body_ptr);
    rm.def("model_ptr", &rbgt::RegionModality::model_ptr);
    rm.def("camera_ptr", &rbgt::RegionModality::camera_ptr);
    rm.def("occlusion_mask_renderer_ptr", &rbgt::RegionModality::occlusion_mask_renderer_ptr);
    // Getters visualization and state
    rm.def("imshow_correspondence", &rbgt::RegionModality::imshow_correspondence);
    rm.def("imshow_pose_update", &rbgt::RegionModality::imshow_pose_update);
    rm.def("imshow_result", &rbgt::RegionModality::imshow_result);
    rm.def("initialized", &rbgt::RegionModality::initialized);

    // -------------------------------------
    // RendererGeometry -> NOT DONE! 
    // 
    // Struct: RenderDataBody
    py::class_<rbgt::RendererGeometry::RenderDataBody, std::shared_ptr<rbgt::RendererGeometry::RenderDataBody>> rdb(m, "RenderDataBody");
    // main methods
    py::class_<rbgt::RendererGeometry, std::shared_ptr<rbgt::RendererGeometry>> rg(m, "RendererGeometry");
    rg.def(py::init<>());
    rg.def("Init", &rbgt::RendererGeometry::Init);
    rg.def("AddBody", &rbgt::RendererGeometry::AddBody);
    rg.def("DeleteBody", &rbgt::RendererGeometry::DeleteBody);
    rg.def("ClearBodies", &rbgt::RendererGeometry::ClearBodies);
    // Getters
    rg.def("render_data_bodies", &rbgt::RendererGeometry::render_data_bodies);
    rg.def("initialized", &rbgt::RendererGeometry::initialized);
    // rg.def("window", &rbgt::RendererGeometry::window); // <- output trype GLFWwindow to be wrapped

    // -------------------------------------
    // Model -> NOT DONE! 
    // 
    // // Struct: PointData
    py::class_<rbgt::Model::PointData, std::shared_ptr<rbgt::Model::PointData>> pd(m, "PointData");
    pd.def_readwrite("center_f_body", &rbgt::Model::PointData::center_f_body);
    pd.def_readwrite("normal_f_body", &rbgt::Model::PointData::normal_f_body);
    pd.def_readwrite("foreground_distance", &rbgt::Model::PointData::foreground_distance);
    pd.def_readwrite("background_distance", &rbgt::Model::PointData::background_distance);
    // // Struct: TemplateView
    py::class_<rbgt::Model::TemplateView, std::shared_ptr<rbgt::Model::TemplateView>> tv(m, "TemplateView");
    tv.def(py::init<>());
    tv.def_readwrite("data_points", &rbgt::Model::TemplateView::data_points);
    tv.def_readwrite("orientation", &rbgt::Model::TemplateView::orientation);
    // constructor
    py::class_<rbgt::Model, std::shared_ptr<rbgt::Model>> model(m, "Model");
    model.def(py::init<std::string&>());
    // model.def("set_name", &rbgt::Model::set_name);
    model.def_readwrite("template_views", &rbgt::Model::template_views_);
    model.def("set_image_size", &rbgt::Model::set_image_size);
    model.def("set_use_random_seed", &rbgt::Model::set_use_random_seed);
    // model.def("set_verbose", &rbgt::Model::set_verbose);
    // // main methods
    model.def("GenerateModel", &rbgt::Model::GenerateModel);
    model.def("LoadModel", &rbgt::Model::LoadModel);
    model.def("SaveModel", &rbgt::Model::SaveModel);
    // model.def("GetClosestTemplateView", &rbgt::Model::GetClosestTemplateView);
    // // Getters
    model.def("name", &rbgt::Model::name);
    model.def("image_size", &rbgt::Model::image_size);
    model.def("use_random_seed", &rbgt::Model::use_random_seed);
    model.def("verbose", &rbgt::Model::verbose);
    model.def("initialized", &rbgt::Model::initialized);

    // // -------------------------------------
    // // Tracker -> NOT DONE! 
    // py::class_<rbgt::Tracker, std::shared_ptr<rbgt::Tracker>> tracker(m, "Tracker");
    // tracker.def(py::init<>());
    // tracker.def("AddRegionModality", &rbgt::Tracker::AddRegionModality);
    // tracker.def("AddViewer", &rbgt::Tracker::AddViewer);
    // tracker.def("set_n_corr_iterations", &rbgt::Tracker::set_n_corr_iterations);
    // tracker.def("set_n_update_iterations", &rbgt::Tracker::set_n_update_iterations);
    // tracker.def("set_visualization_time", &rbgt::Tracker::set_visualization_time);
    // tracker.def("set_viewer_time", &rbgt::Tracker::set_viewer_time);
    // //  Main method
    // tracker.def("StartTracker", &rbgt::Tracker::StartTracker);
    // // Methods for advanced use
    // tracker.def("SetUpObjects", &rbgt::Tracker::SetUpObjects);
    // tracker.def("ExecuteViewingCycle", &rbgt::Tracker::ExecuteViewingCycle);
    // tracker.def("ExecuteTrackingCycle", &rbgt::Tracker::ExecuteTrackingCycle);
    // // Individual steps of tracking cycle for advanced use
    // tracker.def("StartRegionModalities", &rbgt::Tracker::StartRegionModalities);
    // tracker.def("CalculateBeforeCameraUpdate", &rbgt::Tracker::CalculateBeforeCameraUpdate);
    // tracker.def("UpdateCameras", &rbgt::Tracker::UpdateCameras);
    // tracker.def("StartOcclusionMaskRendering", &rbgt::Tracker::StartOcclusionMaskRendering);
    // tracker.def("CalculateCorrespondences", &rbgt::Tracker::CalculateCorrespondences);
    // tracker.def("VisualizeCorrespondences", &rbgt::Tracker::VisualizeCorrespondences);
    // tracker.def("CalculatePoseUpdate", &rbgt::Tracker::CalculatePoseUpdate);
    // tracker.def("VisualizePoseUpdate", &rbgt::Tracker::VisualizePoseUpdate);
    // tracker.def("VisualizeResults", &rbgt::Tracker::VisualizeResults);
    // tracker.def("UpdateViewers", &rbgt::Tracker::UpdateViewers);
    // //  Getters
    // tracker.def("region_modality_ptrs", &rbgt::Tracker::region_modality_ptrs);
    // tracker.def("viewer_ptrs", &rbgt::Tracker::viewer_ptrs);
    // tracker.def("camera_ptrs", &rbgt::Tracker::camera_ptrs);
    // tracker.def("occlusion_mask_renderer_ptrs", &rbgt::Tracker::occlusion_mask_renderer_ptrs);
    // tracker.def("n_corr_iterations", &rbgt::Tracker::n_corr_iterations);
    // tracker.def("n_update_iterations", &rbgt::Tracker::n_update_iterations);
    // tracker.def("visualization_time", &rbgt::Tracker::visualization_time);
    // tracker.def("viewer_time", &rbgt::Tracker::viewer_time);

    // -------------------------------------
    // NormalImageRenderer -> NOT DONE! 
    py::class_<rbgt::NormalImageRenderer, rbgt::Renderer, std::shared_ptr<rbgt::NormalImageRenderer>> nir(m, "NormalImageRenderer");
    nir.def(py::init<>());
    nir.def("Init", &rbgt::NormalImageRenderer::Init);
    nir.def("set_intrinsics", &rbgt::NormalImageRenderer::set_intrinsics);
    nir.def("set_z_min", &rbgt::NormalImageRenderer::set_z_min);
    nir.def("set_z_max", &rbgt::NormalImageRenderer::set_z_max);
    nir.def("set_depth_scale", &rbgt::NormalImageRenderer::set_depth_scale);
    // Main method
    nir.def("StartRendering", &rbgt::NormalImageRenderer::StartRendering);
    nir.def("FetchNormalImage", &rbgt::NormalImageRenderer::FetchNormalImage);
    nir.def("FetchDepthImage", &rbgt::NormalImageRenderer::FetchDepthImage);
    // Getters for images and internal variables
    nir.def("normal_image", &rbgt::NormalImageRenderer::normal_image);
    nir.def("depth_image", &rbgt::NormalImageRenderer::depth_image);
    nir.def("depth_scale", &rbgt::NormalImageRenderer::depth_scale);
    // Getter that calculates a point vector based on a rendered depth image
    nir.def("GetPointVector", &rbgt::NormalImageRenderer::GetPointVector); // <- input + output not wrapped

    // Data structures

    // -------------------------------------
    // Transform3fA, vector
    py::class_<Transform3fA> t3fa(m, "Transform3fA");
    t3fa.def(py::init<>());

    py::class_<Eigen::Vector3f> v3f(m, "Vector3f");
    py::class_<cv::Vec3b> v3b(m, "Vec3b");

    // -------------------------------------
    // intrinstics
    py::class_<rbgt::Intrinsics> intrinsitcs(m, "Intrinsics");
    intrinsitcs.def(py::init<>());
    intrinsitcs.def_readwrite("fu", &rbgt::Intrinsics::fu);
    intrinsitcs.def_readwrite("fv", &rbgt::Intrinsics::fv);
    intrinsitcs.def_readwrite("ppu", &rbgt::Intrinsics::ppu);
    intrinsitcs.def_readwrite("ppv", &rbgt::Intrinsics::ppv);
    intrinsitcs.def_readwrite("width", &rbgt::Intrinsics::width);
    intrinsitcs.def_readwrite("height", &rbgt::Intrinsics::height);

    // // -------------------------------------
    // // GLFWwindow
    // py::class_<GLFWwindow> glfwWindow(m, "GLFWwindow");
    // glfwWindow.def(py::init<>());

    // ----------------------------------
    // path2filesystem::path converter
    m.def("filesystem", &filesystem);
    m.def("fs2string", &fs2string);
    py::class_<std::experimental::filesystem::path> fs(m, "FileSystem");

    // ----------------------------------
    // len 12 matrix to t3fA converter
    m.def("mat2t3fA", &mat2t3fA);
    m.def("mat2t3fAfullMat", &mat2t3fAfullMat);
    m.def("t3fA2mat", &t3fA2mat);
    m.def("vector3f2list", &vector3f2list);
    m.def("vector3b2list", &vector3b2list);
    m.def("list2cv2i", &list2cv2i);
    m.def("mat_to_nparray", &mat_to_nparray);
    m.def("nparray_to_mat_uint8", &nparray_to_mat_uint8);
    m.def("mat_to_nparray_clone", &mat_to_nparray_clone);

    // closest template view converter
    m.def("getClosestTemplateView", &getClosestTemplateView);

    // ----------------------------------
    // cv::Mat
    py::class_<cv::Mat> mat(m, "cvMat");
    mat.def_readwrite("cols", &cv::Mat::cols);
    mat.def_readwrite("rows", &cv::Mat::rows);
    py::class_<cv::Point2i> cv2i(m, "cvPoint2i");
    cv2i.def_readwrite("x", &cv::Point2i::x);
    cv2i.def_readwrite("y", &cv::Point2i::y);
    // mat.def("at", &cv::Mat::at<cv::Vec3b>);
    // mat.def("at", py::overload_cast<int, int>(&cv::Mat::at));
}

