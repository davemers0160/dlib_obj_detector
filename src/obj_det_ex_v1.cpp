#define _CRT_SECURE_NO_WARNINGS

// #if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
// #include <windows.h>
// #endif

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <thread>
#include <sstream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <string>
#include <utility>
#include <stdexcept>

// Custom includes
#include "obj_det_dnn.h"
#include "get_platform.h"
//#include "file_parser.h"
#include "get_current_time.h"

//#include "gorgon_capture.h"

#include "num2string.h"
//#include "array_image_operations.h"

// Net Version
#include "yj_net_v4.h"
#include "load_data.h"
#include "eval_net_performance.h"
//#include "enhanced_array_cropper.h"
//#include "random_channel_swap.h"
//#include "enhanced_channel_swap.h"


// dlib includes
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>
#include <dlib/rand.h>

// dlib-contrib includes
#include "array_image_operations.h"
#include "random_array_cropper.h"


// new copy and set learning rate includes
//#include "copy_dlib_net.h"
//#include "dlib_set_learning_rates.h"

// -------------------------------GLOBALS--------------------------------------

extern const uint32_t array_depth;
std::string platform;

//this will store the standard RGB images and groundtruth data for the bounding box labels
//std::vector<dlib::matrix<dlib::rgb_pixel>> train_images, test_images;
std::vector<std::array<dlib::matrix<uint8_t>, array_depth>> train_images, test_images;
//std::array<dlib::matrix<uint8_t>, array_depth> train_image, test_image;
//std::vector<dlib::mmod_rect> train_label, test_label;
std::vector<std::vector<dlib::mmod_rect>> train_labels, test_labels;

// containers to store the random crops used during each training iteration and groundtruth data for the bounding box labels
std::vector<std::array<dlib::matrix<uint8_t>, array_depth>> train_batch_samples, test_batch_samples;
std::vector<std::vector<dlib::mmod_rect>> train_batch_labels, test_batch_labels;

std::string version;
std::string net_name = "obj_det_net_";
std::string net_sync_name = "obj_det_sync_";
std::string logfileName = "obj_det_net_log_";
//std::string gorgon_savefile = "gorgon_obj_det_";

// ----------------------------------------------------------------------------
void get_platform_control(void)
{
    get_platform(platform);

    if (platform == "")
    {
        std::cout << "No Platform could be identified... defaulting to Windows." << std::endl;
        platform = "Win";
    }

    version = version + platform;
    net_sync_name = version + "_sync";
    logfileName = version + "_log_";
    net_name = version +  "_final_net.dat";
}

// ----------------------------------------------------------------------------------------

void print_usage(void)
{
    std::cout << "Enter the following as arguments into the program:" << std::endl;
    std::cout << "<image file name> " << std::endl;
    std::cout << endl;
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{

    uint64_t idx = 0, jdx = 0;
    uint8_t HPC = 0;
    std::string sdate, stime;

    // data IO variables
    const std::string os_file_sep = "/";
    std::string program_root;
    std::string save_directory;
    std::string sync_save_location;
    std::string image_save_location;
    std::string results_save_location;
    std::string train_inputfile;
    std::string test_inputfile;
    std::string train_data_directory, test_data_directory;
    std::vector<std::vector<std::string>> training_file;
    std::vector<std::vector<std::string>> test_file;
    std::vector<std::string> tr_image_files, te_image_files;
    std::ofstream DataLogStream;

    // timing variables
    typedef std::chrono::duration<double> d_sec;
    auto start_time = chrono::system_clock::now();
    auto stop_time = chrono::system_clock::now();
    auto elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

    // training variables
    int32_t stop = -1;
    std::vector<std::string> stop_codes = { "Minimum Learning Rate Reached.", "Max Training Time Reached", "Max Training Steps Reached" };
    std::vector<double> stop_criteria;
    //uint64_t num_crops;
    //std::pair<uint64_t, uint64_t> crop_size;
    //std::pair<double, double> angles = std::make_pair(15.0, 15.0);
    training_params tp;
    std::vector<uint32_t> filter_num;

    crop_info ci;

    std::pair<uint32_t, uint32_t> target_size;  // min_target_size, max_target_size

    std::vector<int32_t> gpu;
    uint64_t one_step_calls = 0;
    uint64_t epoch = 0;
    uint64_t index = 0;   

    //create window to display images
    dlib::image_window win;
    dlib::rgb_pixel color;
    dlib::matrix<dlib::rgb_pixel> rgb_img;

    // set the learning rate multipliers: 0 means freeze the layers; r1 = learning rate multiplier, r2 = learning rate bias multiplier
    //double r1 = 1.0, r2 = 1.0;
    
    // ----------------------------------------------------------------------------------------
   
    if (argc == 1)
    {
        print_usage();
        std::cin.ignore();
        return 0;
    }

    std::string parse_filename = argv[1];

    // parse through the supplied csv file
    parse_input_file(parse_filename, version, gpu, stop_criteria, tp, train_inputfile, test_inputfile, ci, target_size, filter_num, save_directory);

    // check the platform
    get_platform_control();

    // check for HPC <- set the environment variable PLATFORM to HPC
    if(platform.compare(0,3,"HPC") == 0)
    {
        std::cout << "HPC Platform Detected." << std::endl;
        HPC = 1;
    }

    // setup save variable locations
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
    //program_root = get_path(get_path(get_path(std::string(argv[0]), "\\"), "\\"), "\\") + os_file_sep;
    sync_save_location = save_directory + "nets/";
    results_save_location = save_directory + "results/";
    image_save_location = save_directory + "result_images/";

#else
    if (HPC == 1)
    {
        //HPC version
        program_root = get_path(get_path(get_path(std::string(argv[0]), os_file_sep), os_file_sep), os_file_sep) + os_file_sep;
    }
    else
    {
        // Ubuntu
        if(platform.compare(0,8,"MainGear") == 0)
        {
            program_root = "/home/owner/Projects/machineLearningResearch/";
        }
        else
        {
            if (platform.compare(0,7,"SL02319") == 0)
            {
                // fill in the location of where the root program is running
                program_root = "/media/daleas/DATA/Ashley_ML/machineLearningResearch/";
            }
            else
            {
                // fill in the location of where the root program is running
                program_root = "/mnt/data/machineLearningResearch/";
            }

        }
    }

    sync_save_location = save_directory + "nets/";
    results_save_location = save_directory + "results/";
    image_save_location = save_directory + "result_images/";

#endif

    std::cout << "Reading Inputs... " << std::endl;
    std::cout << "Platform:              " << platform << std::endl;
    std::cout << "GPU:                   { ";
    for (idx = 0; idx < gpu.size(); ++idx)
        std::cout << gpu[idx] << " ";
    std::cout << "}" << std::endl;
    //std::cout << "program_root:          " << program_root << std::endl;
    std::cout << "save_directory:        " << save_directory << std::endl;
    std::cout << "sync_save_location:    " << sync_save_location << std::endl;
    std::cout << "results_save_location: " << results_save_location << std::endl;
    std::cout << "image_save_location:   " << image_save_location << std::endl;


    try {

        get_current_time(sdate, stime);
        logfileName = logfileName + sdate + "_" + stime + ".txt";
        //cropper_stats_file = output_save_location + "cr_stats_" + version + "_" + sdate + "_" + stime + ".txt";

        std::cout << "Log File:              " << (results_save_location + logfileName) << std::endl << std::endl;
        DataLogStream.open((results_save_location + logfileName), ios::out | ios::app);

        // Add the date and time to the start of the log file
        DataLogStream << "------------------------------------------------------------------" << std::endl;
        DataLogStream << "Version: 2.0    Date: " << sdate << "    Time: " << stime << std::endl;
        DataLogStream << "Platform: " << platform << std::endl;
        DataLogStream << "GPU: { ";
        for (idx = 0; idx < gpu.size(); ++idx)
            DataLogStream << gpu[idx] << " ";
        DataLogStream << "}" << std::endl;
        DataLogStream << "------------------------------------------------------------------" << std::endl;

//-----------------------------------------------------------------------------
// Read in the training and testing images
//-----------------------------------------------------------------------------

        // parse through the supplied training csv file
        parse_group_csv_file(train_inputfile, '{', '}', training_file);
        if (training_file.size() == 0)
        {
            throw std::runtime_error("Training file is empty");
        }

        // the first line in this file is now the data directory
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
        train_data_directory = training_file[0][0];
#else
        if (HPC == 1)
        {
            train_data_directory = training_file[0][3];
        }
        else if (platform.compare(0,7,"SL02319") == 0)
        {
            train_data_directory = training_file[0][2];
        }
        else
        {
            train_data_directory = training_file[0][1];
        }
#endif

        training_file.erase(training_file.begin());

        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "data_directory:        " << train_data_directory << std::endl;

        std::cout << train_inputfile << std::endl;
        std::cout << "Training image sets to parse: " << training_file.size() << std::endl;

        DataLogStream << train_inputfile << std::endl;
        DataLogStream << "Training image sets to parse: " << training_file.size() << std::endl;
        
        std::cout << "Loading training images... ";

        // load in the images and labels
        start_time = chrono::system_clock::now();
        load_data(training_file, train_data_directory, train_images, train_labels, tr_image_files);
        stop_time = chrono::system_clock::now();
        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

        std::cout << "Loaded " << train_images.size() << " training image sets in " << elapsed_time.count() / 60 << " minutes." << std::endl;
        DataLogStream << "Loaded " << train_images.size() << " training image sets in " << elapsed_time.count() / 60 << " minutes." << std::endl;
        
        //DataLogStream << "the following objects were ignored: " << std::endl << std::endl;

        //int num_ignored_train_images = 0;
        //int num_found_train_images = 0;

        //for (int idx = 0; idx < train_labels.size(); ++idx) 
        //{
        //    for (int jdx = 0; jdx < train_labels[idx].size(); ++jdx) 
        //    {
        //        if (train_labels[idx][jdx].ignore == true) 
        //        {
        //            ++num_ignored_train_images;
        //            DataLogStream << training_file[idx][1] << " " << training_file[idx][6] << std::endl;
        //        }
        //        else 
        //        {
        //            ++num_found_train_images;
        //        }
        //    }
        //}

        //std::cout << "Number of Found Train Objects: " << num_found_train_images << std::endl;
        //std::cout << "Number of Ignored Train Objects : " << num_ignored_train_images << std::endl;
        //std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

        //DataLogStream << "Number of Found Train Objects: " << num_found_train_images << std::endl;
        //DataLogStream << "Number of Ignored Train Objects: " << num_ignored_train_images << std::endl<<std::endl;



        // ------------------------------------------------------------------------------------

        // for debugging to view the images
        //for (idx = 0; idx < training_file.size(); ++idx)
        //{   

        //    win.clear_overlay();
        //    win.set_image(train_images[idx]);

        //    for (jdx = 0; jdx < train_labels[idx].size(); ++jdx)
        //    {
        //        color = train_labels[idx][jdx].ignore ? dlib::rgb_pixel(0, 0, 255) : dlib::rgb_pixel(0, 255, 0);
        //        win.add_overlay(train_labels[idx][jdx].rect, color);
        //    }

        //    win.set_title(("Training Image: " + num2str(idx+1,"%05u")));

        //    //std::cin.ignore();
        //    dlib::sleep(500);
        //}


        //-------------------------------------------------------------------------------------
        // parse through the supplied test csv file
        // parseCSVFile(test_inputfile, test_file);
        parse_group_csv_file(test_inputfile, '{', '}', test_file);
        if (test_file.size() == 0)
        {
            throw std::runtime_error("Test file is empty");
        }

        // the data directory should be the first entry in the input file
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
        test_data_directory = test_file[0][0];
#else
        if (HPC == 1)
        {
            test_data_directory = test_file[0][2];
        }
        else if (platform.compare(0,7,"SL02319") == 0)
        {
            test_data_directory = test_file[0][2];
        }
        else
        {
            test_data_directory = test_file[0][1];
        }
#endif

        test_file.erase(test_file.begin());
        std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
        std::cout << "data_directory:        " << test_data_directory << std::endl;
        std::cout << test_inputfile << std::endl;
        std::cout << "Test image sets to parse: " << test_file.size() << std::endl;

        DataLogStream << "------------------------------------------------------------------" << std::endl;
        DataLogStream << test_inputfile << std::endl;
        DataLogStream << "Test image sets to parse: " << test_file.size() << std::endl;
       
        std::cout << "Loading test images... ";

        // load in the images and labels
        start_time = chrono::system_clock::now();
        load_data(test_file, test_data_directory, test_images, test_labels, te_image_files);
        stop_time = chrono::system_clock::now();
        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

        std::cout << "Loaded " << test_images.size() << " test image sets in " << elapsed_time.count() / 60 << " minutes." << std::endl;
        DataLogStream << "Loaded " << test_images.size() << " test image sets in " << elapsed_time.count() / 60 << " minutes." << std::endl;
        
        //DataLogStream << "the following objects were ignored: " << std::endl << std::endl;

        //for (int idx = 0; idx < test_labels.size(); ++idx) 
        //{
        //    for (int jdx = 0; jdx < test_labels[idx].size(); ++jdx) 
        //    {
        //        if (test_labels[idx][jdx].ignore == true) 
        //        {
        //            ++num_ignored_test_images;
        //            DataLogStream << test_file[idx][1] << " " << test_file[idx][6] << std::endl;
        //        }
        //        else 
        //        {
        //            ++num_found_test_images;
        //        }
        //    }
        //}

        //std::cout << "Number of Found Test Objects : " << num_found_test_images << std::endl;
        //std::cout << "Number of Ignored Test Objects : " << num_ignored_test_images << std::endl;
        //std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

        //DataLogStream << "Number of Found Test Objects : " << num_found_test_images << std::endl;
        //DataLogStream << "Number of Ignored Test Objects: " << num_ignored_test_images << std::endl << std::endl;
        //DataLogStream << "------------------------------------------------------------------" << std::endl<<std::endl;

        // ------------------------------------------------------------------------------------

        // for debugging to view the images
        //for (idx = 0; idx < test_images.size(); ++idx)
        //{

        //    win.clear_overlay();
        //    win.set_image(test_images[idx]);

        //    for (jdx = 0; jdx < test_labels[idx].size(); ++jdx)
        //    {
        //        color = test_labels[idx][jdx].ignore ? dlib::rgb_pixel(0, 0, 255) : dlib::rgb_pixel(0, 255, 0);
        //        win.add_overlay(test_labels[idx][jdx].rect, color);
        //    }

        //    win.set_title(("Training Image: " + num2str(idx+1,"%05u")));

        //    std::cin.ignore();
        //    //dlib::sleep(800);
        //}

//-----------------------------------------------------------------------------
// Setup the network
//-----------------------------------------------------------------------------

        // this sets th GPUs to use algorithms that are smaller in memory but may take a little longer to execute
        dlib::set_dnn_prefer_smallest_algorithms();

        // set the cuda device explicitly
        if (gpu.size() == 1)
            dlib::cuda::set_device(gpu[0]);

        // For further details see the mmod_options documentation.
        dlib::mmod_options options(train_labels, target_size.second, target_size.first, 0.75);

        // example of how to push back a custion window
        // options.detector_windows.push_back(dlib::mmod_options::detector_window_details(114, 103));

        options.loss_per_false_alarm = 1.0;
        options.loss_per_missed_target = 2.0;
        options.truth_match_iou_threshold = 0.40;
        options.overlaps_nms = dlib::test_box_overlap(0.4, 1.0);

        std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
        DataLogStream << std::endl << "------------------------------------------------------------------" << std::endl;

        std::cout << "num detector windows: " << options.detector_windows.size() << std::endl;
        DataLogStream << "num detector windows: " << options.detector_windows.size() << std::endl;

        for (auto& w : options.detector_windows)
        {
            std::cout << "detector window (w x h): " << w.label << " - " << w.width << " x " << w.height << std::endl;
            DataLogStream << "detector window (w x h): " << w.label << " - " << w.width << " x " << w.height << std::endl;
        }
        
        std::cout << "overlap NMS IOU thresh:             " << options.overlaps_nms.get_iou_thresh() << std::endl;
        std::cout << "overlap NMS percent covered thresh: " << options.overlaps_nms.get_percent_covered_thresh() << std::endl;

        DataLogStream << "overlap NMS IOU thresh:             " << options.overlaps_nms.get_iou_thresh() << std::endl;
        DataLogStream << "overlap NMS percent covered thresh: " << options.overlaps_nms.get_percent_covered_thresh() << std::endl;
        DataLogStream << std::endl << "------------------------------------------------------------------"  << std::endl;


        // Now we are ready to create our network and trainer.
//#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
        //net_type net;

        // load in the convolutional filter numbers from the input file
        //config_net(net, options, filter_num);
        net_type net = config_net<net_type>(options, filter_num);

//#else
        // check for the gcc version
        //#if defined(__GNUC__) && (__GNUC__ > 5)
        //    net_type net(options);
            //config_net(net, options, filter_num);
        //#else
        //    net_type net;
        //    config_net(net, options, filter_num);
        //#endif
//#endif

        // The MMOD loss requires that the number of filters in the final network layer equal
        // options.detector_windows.size().  So we set that here as well.
        net.subnet().layer_details().set_num_filters(options.detector_windows.size());

        dlib::dnn_trainer<net_type, dlib::adam> trainer(net, dlib::adam(0.0001, 0.9, 0.99),  gpu);
        trainer.set_learning_rate(tp.intial_learning_rate);
        trainer.be_verbose();
        trainer.set_synchronization_file((sync_save_location + net_sync_name), std::chrono::minutes(5));
        trainer.set_iterations_without_progress_threshold(tp.steps_wo_progess);
        trainer.set_test_iterations_without_progress_threshold(5000);
        trainer.set_learning_rate_shrink_factor(tp.learning_rate_shrink_factor);

        // set the batch normalization stats window to something big
        dlib::set_all_bn_running_stats_window_sizes(net, 1000);

        dlib::random_array_cropper cropper;

        cropper.set_seed(time(NULL));

        // set the rows, cols for the cropped image size
        cropper.set_chip_dims(ci.crop_height, ci.crop_width);

        // Usually you want to give the cropper whatever min sizes you passed to the
        // mmod_options constructor, which is what we do here.
        cropper.set_min_object_size(target_size.second+2, target_size.first+2);   // plane

        cropper.set_max_object_size(1.0);   // 0.8

        // percetange of crops that don't contain an object of interest
        cropper.set_background_crops_fraction(0.4);

        // randomly flip left-right
        cropper.set_randomly_flip(true);

        // maximum allowed rotation +/-
        cropper.set_max_rotation_degrees(ci.angle);

        // set the cropper stats recorder
        //cropper.set_stats_filename(cropper_stats_file);

        dlib::rand rnd;
        rnd = dlib::rand(time(NULL));

        // display a few hits and also save them to the log file for later analysis
        std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
        std::cout << "Crop count: " << ci.crop_num << std::endl;
        DataLogStream << "Crop count: " << ci.crop_num << std::endl;

        // show all of the cropper settings
        std::cout << cropper << std::endl;
        DataLogStream << cropper << std::endl;
        DataLogStream << "------------------------------------------------------------------" << std::endl;

        // show all of the trainer settings
        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << trainer << std::endl;
        DataLogStream << trainer << std::endl;
        DataLogStream << "------------------------------------------------------------------" << std::endl;

        // show the network to verify that it looks correct
        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Net Name: " << net_name << std::endl;
        std::cout << net << std::endl;

        DataLogStream << "Net Name: " << net_name << std::endl;
        DataLogStream << net << std::endl;
        DataLogStream << "------------------------------------------------------------------" << std::endl;

//-----------------------------------------------------------------------------
// TRAINING START
//-----------------------------------------------------------------------------

        // these two matrices will contain the results of the training and testing
        dlib::matrix<double, 1, 6> training_results = dlib::zeros_matrix<double>(1, 6);
        dlib::matrix<double, 1, 6> test_results = dlib::zeros_matrix<double>(1, 6);

        double train_lr = trainer.get_learning_rate();

        uint64_t test_step_count = 100;

        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Starting Training..." << std::endl;
        start_time = chrono::system_clock::now();

        while(stop < 0)
        {
            // first check to make sure that the final_learning_rate hasn't been exceeded
            if (trainer.get_learning_rate() >= tp.final_learning_rate)
            {
                //cropper.file_append(num_crops, train_data_directory, training_file, mini_batch_samples, mini_batch_labels);
                cropper(ci.crop_num, train_images, train_labels, train_batch_samples, train_batch_labels);

#if defined(_DEBUG)
/*                
                for (idx = 0; idx < train_batch_samples.size(); ++idx)
                {

                    merge_channels(train_batch_samples[idx], rgb_img);
                    win.clear_overlay();
                    win.set_image(rgb_img);

                    for (jdx = 0; jdx < train_batch_labels[idx].size(); ++jdx)
                    {
                        color = train_batch_labels[idx][jdx].ignore ? dlib::rgb_pixel(0, 0, 255) : dlib::rgb_pixel(0, 255, 0);
                        win.add_overlay(train_batch_labels[idx][jdx].rect, color);
                    }
                    std::cin.ignore();
                }
*/
#endif

                trainer.train_one_step(train_batch_samples, train_batch_labels);

            }
            else
            {
                stop = 0;
            }

            one_step_calls = trainer.get_train_one_step_calls();

            if((one_step_calls % test_step_count) == 0)
            {
                // this is where we will perform any needed evaluations of the network
                // detction_accuracy, correct_hits, false_positives, missing_detections

                cropper(ci.crop_num, test_images, test_labels, test_batch_samples, test_batch_labels);

                trainer.test_one_step(test_batch_samples, test_batch_labels);

                test_results = dlib::zeros_matrix<double>(1, 6);

/*
                for (idx = 0; idx < test_file.size(); ++idx)
                {    
                    test_label.clear();
                    load_single_set(test_data_directory, test_file[idx], test_image, test_label);

                    merge_channels(test_image, rgb_img);
                    //std::cout << te_image_files[idx].first;
                    //win.clear_overlay();
                    //win.set_image(rgb_img);
                    v_win[idx].clear_overlay();
                    v_win[idx].set_image(rgb_img);
                 
                    //v_win[idx].clear_overlay();
                    //v_win[idx].set_image(tmp_img);

                    std::vector<dlib::mmod_rect> dnn_labels;           

                    // get the rough classification time per image
                    start_time = chrono::system_clock::now();
                    dlib::matrix<double, 1, 6> tr = eval_net_performance(net, test_image, test_label, dnn_labels, min_target_size, fda_test_box_overlap(0.3, 1.0));
                    stop_time = chrono::system_clock::now();

                    elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);
                    dnn_test_labels.push_back(dnn_labels);

                    //overlay the dnn detections on the image
                    for (jdx = 0; jdx < dnn_labels.size(); ++jdx)
                    {
                    v_win[idx].add_overlay(dnn_labels[jdx].rect, dlib::rgb_pixel(255, 0, 0));
                    //draw_rectangle(tmp_img, dnn_labels[jdx].rect, dlib::rgb_pixel(255, 0, 0), 2);
                    //DataLogStream << "Detect Confidence Level (" << dnn_test_labels[idx][jdx].label << "): " << dnn_test_labels[idx][jdx].detection_confidence << std::endl;
                    //std::cout << "Detect Confidence Level (" << dnn_test_labels[idx][jdx].label << "): " << dnn_test_labels[idx][jdx].detection_confidence << std::endl;
                    }
                    
                    std::cout << ".";
                    // overlay the ground truth boxes on the image
                    for (jdx = 0; jdx < test_label.size(); ++jdx)
                    {
                        v_win[idx].add_overlay(test_label[jdx].rect, dlib::rgb_pixel(0, 255, 0));
                        draw_rectangle(rgb_img, test_label[jdx].rect, dlib::rgb_pixel(0, 255, 0), 2);
                    }
                    
                    //save results in image form
                    //std::string image_save_name = output_save_location + "test_save_image_" + version + num2str(idx, "_%03d.png");
                    //save_png(rgb_img, image_save_name);
                    std::cout << std::endl;

                    test_results += tr;
                }

                test_results(0, 0) = test_results(0, 0) / (double)test_file.size();

                std::cout << "------------------------------------------------------------------" << std::endl;
                std::cout << "Results (DA, CH, FP, MD): " << std::fixed << std::setprecision(4) << test_results(0, 0) << ", " << test_results(0, 3) << ", " << test_results(0, 4) << ", " << test_results(0, 5) << std::endl;
                std::cout << "------------------------------------------------------------------" << std::endl;
*/                

                DataLogStream << std::setw(6) << std::setfill('0') << one_step_calls << ", " << std::fixed << std::setprecision(9) << trainer.get_learning_rate() << ", ";
                DataLogStream << std::setprecision(5) << trainer.get_average_loss() << ", " << trainer.get_average_test_loss() << std::endl;

            }

            // now check to see if we've trained long enough according to the input time limit
            stop_time = chrono::system_clock::now();
            elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

            if((double)elapsed_time.count()/(double)3600.0 > stop_criteria[0])
            {
                stop = 1;
            }

            // finally check to see if we've exceeded the max number of one step training calls
            // according to the input file
            if (one_step_calls >= stop_criteria[1])
            {
                stop = 2;
            }

        }   // end of while(stop<0)


//-----------------------------------------------------------------------------
// TRAINING STOP
//-----------------------------------------------------------------------------

        stop_time = chrono::system_clock::now();
        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

        // wait for training threads to stop
        trainer.get_net();

        std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
        std::cout << "Elapsed Training Time: " << elapsed_time.count() / 3600 << " hours" << std::endl;
        std::cout << "Stop Code: " << stop_codes[stop] << std::endl;
        std::cout << "Final Average Loss: " << trainer.get_average_loss() << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

        DataLogStream << "------------------------------------------------------------------" << std::endl;
        DataLogStream << "Elapsed Training Time: " << elapsed_time.count() / 3600 << " hours" << std::endl;
        DataLogStream << "Stop Code: " << stop_codes[stop] << std::endl;
        DataLogStream << "Final Average Loss: " << trainer.get_average_loss() << std::endl;
        DataLogStream << "------------------------------------------------------------------" << std::endl << std::endl;

        // Save the network to disk
        net.clean();
        dlib::serialize(sync_save_location + net_name) << net;

//-----------------------------------------------------------------------------
//          EVALUATE THE FINAL NETWORK PERFORMANCE
//-----------------------------------------------------------------------------

        // load the network from the saved file
        anet_type test_net;

        std::cout << std::endl << "Loading " << (sync_save_location + net_name) << std::endl;
        dlib::deserialize(sync_save_location + net_name) >> test_net;


        // In this section we want to evaluate the performance of the network against the training data
        // this should be displayed and then saved to the log file
        // - This can also include displaying the input image along with the ground truth bounding box, name and dnn results
        
        //---------------------------------------------------------------------------------------
        std::cout << "Analyzing Training Results..." << std::endl;

        training_results = dlib::zeros_matrix<double>(1, 6);

        for (idx = 0; idx < train_images.size(); ++idx)
        {

            merge_channels(train_images[idx], rgb_img);
            win.clear_overlay();
            win.set_image(rgb_img);

            std::vector<dlib::mmod_rect> dnn_labels;

            // get the rough classification time per image
            start_time = chrono::system_clock::now();
            dlib::matrix<double, 1, 6> tr = eval_net_performance(test_net, train_images[idx], train_labels[idx], dnn_labels, target_size.first, fda_test_box_overlap(0.4, 1.0));
            stop_time = chrono::system_clock::now();

            elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

            std::cout << "------------------------------------------------------------------" << std::endl;
            std::cout << "Image " << std::setw(5) << std::setfill('0') << idx << ": " << tr_image_files[idx] << std::endl;
            std::cout << "Image Size (h x w): " << train_images[idx][0].nr() << "x" << train_images[idx][0].nc() << std::endl;
            std::cout << "Classification Time (s): " << elapsed_time.count() << std::endl;
            std::cout << "Results: " << std::fixed << std::setprecision(4) << tr(0, 0) << ", " << tr(0, 3) << ", " << tr(0, 4) << ", " << tr(0, 5) << std::endl;

            DataLogStream << "------------------------------------------------------------------" << std::endl;
            DataLogStream << "Image " << std::setw(5) << std::setfill('0') << idx << ": " << tr_image_files[idx] << std::endl;
            DataLogStream << "Image Size (h x w): " << train_images[idx][0].nr() << "x" << train_images[idx][0].nc() << std::endl;
            DataLogStream << "Classification Time (s): " << elapsed_time.count() << std::endl;
            DataLogStream << "Results: " << std::fixed << std::setprecision(4) << tr(0, 0) << ", " << tr(0, 3) << ", " << tr(0, 4) << ", " << tr(0, 5) << std::endl;

            //overlay the dnn detections on the image
			std::string results = num2str(idx, "%04d");
            for (jdx = 0; jdx < dnn_labels.size(); ++jdx)
            {
                win.add_overlay(dnn_labels[jdx].rect, dlib::rgb_pixel(255, 0, 0));
                draw_rectangle(rgb_img, dnn_labels[jdx].rect, dlib::rgb_pixel(255, 0, 0),2);
                DataLogStream << "Detect Confidence Level (" << dnn_labels[jdx].label << "): " << dnn_labels[jdx].detection_confidence << std::endl;
                std::cout << "Detect Confidence Level (" << dnn_labels[jdx].label << "): " << dnn_labels[jdx].detection_confidence << std::endl;
				results = results + ",{" + num2str(dnn_labels[jdx].rect.left(),"%d,") + num2str(dnn_labels[jdx].rect.top(),"%d,") + num2str(dnn_labels[jdx].rect.width(),"%d,") + num2str(dnn_labels[jdx].rect.height(),"%d,") + dnn_labels[jdx].label + "},";
            }
			DataLogStream << results.substr(0,results.length()-2) << std::endl; 

            // overlay the ground truth boxes on the image
            for (jdx = 0; jdx < train_labels[idx].size(); ++jdx)
            {
                win.add_overlay(train_labels[idx][jdx].rect, dlib::rgb_pixel(0, 255, 0));
                draw_rectangle(rgb_img, train_labels[idx][jdx].rect, dlib::rgb_pixel(0,255,0),2);
            }

            //save results in image form
            //std::string image_save_name = output_save_location + "train_save_image_" + version + num2str(idx, "_%03d.png");
            //save_png(tmp_img, image_save_name);

            training_results += tr;
            //dlib::sleep(50);
            //std::cin.ignore();

        }

        DataLogStream << "------------------------------------------------------------------" << std::endl << std::endl;

        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Training Results (detction_accuracy, correct_hits, false_positives, missing_detections): " << std::fixed << std::setprecision(4) << training_results(0, 0) / (double)training_file.size() << ", " << training_results(0, 3) << ", " << training_results(0, 4) << ", " << training_results(0, 5) << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

        // In this section we want to evaluate the performance of the network against the test data
        // this should be displayed and then saved to the log file
        // - This can also include displaying the input image along with the ground truth bounding box, name and dnn results
        std::cout << "Analyzing Test Results..." << std::endl;

        // testResults = eval_all_net_performance(test_net, test_images, test_labels, dnn_test_labels, min_target_size);
        test_results = dlib::zeros_matrix<double>(1, 6);
        //dnn_test_labels.clear();

        for (idx = 0; idx < test_images.size(); ++idx)
        {
            //test_label.clear();
            //load_single_set(test_data_directory, test_file[idx], test_image, test_label);

            merge_channels(test_images[idx], rgb_img);
            //std::cout << te_image_files[idx].first;
            win.clear_overlay();
            win.set_image(rgb_img);

            std::vector<dlib::mmod_rect> dnn_labels;

            // get the rough classification time per image
            start_time = chrono::system_clock::now();
            dlib::matrix<double, 1, 6> tr = eval_net_performance(test_net, test_images[idx], test_labels[idx], dnn_labels, target_size.first, fda_test_box_overlap(0.4, 1.0));
            stop_time = chrono::system_clock::now();

            elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

            std::cout << "------------------------------------------------------------------" << std::endl;
            std::cout << "Image " << std::setw(5) << std::setfill('0') << idx << ": " << te_image_files[idx] << std::endl;
            std::cout << "Image Size (h x w): " << test_images[idx][0].nr() << "x" << test_images[idx][0].nc() << std::endl;
            std::cout << "Classification Time (s): " << elapsed_time.count() << std::endl;
            std::cout << "Results: " << std::fixed << std::setprecision(4) << tr(0, 0) << ", " << tr(0, 3) << ", " << tr(0, 4) << ", " << tr(0, 5) << std::endl;

            DataLogStream << "------------------------------------------------------------------" << std::endl;
            DataLogStream << "Image " << std::setw(5) << std::setfill('0') << idx << ": " << te_image_files[idx] << std::endl;
            DataLogStream << "Image Size (h x w): " << test_images[idx][0].nr() << "x" << test_images[idx][0].nc() << std::endl;
            DataLogStream << "Classification Time (s): " << elapsed_time.count() << std::endl;
            DataLogStream << "Results: " << std::fixed << std::setprecision(4) << tr(0, 0) << ", " << tr(0, 3) << ", " << tr(0, 4) << ", " << tr(0, 5) << std::endl;

            //overlay the dnn detections on the image
            for (jdx = 0; jdx < dnn_labels.size(); ++jdx)
            {
                win.add_overlay(dnn_labels[jdx].rect, dlib::rgb_pixel(255, 0, 0));
                draw_rectangle(rgb_img, dnn_labels[jdx].rect, dlib::rgb_pixel(255, 0, 0), 2);
                DataLogStream << "Detect Confidence Level (" << dnn_labels[jdx].label << "): " << dnn_labels[jdx].detection_confidence << std::endl;
                std::cout << "Detect Confidence Level (" << dnn_labels[jdx].label << "): " << dnn_labels[jdx].detection_confidence << std::endl;
            }

            // overlay the ground truth boxes on the image
            for (jdx = 0; jdx < test_labels[idx].size(); ++jdx)
            {
                win.add_overlay(test_labels[idx][jdx].rect, dlib::rgb_pixel(0, 255, 0));
                draw_rectangle(rgb_img, test_labels[idx][jdx].rect, dlib::rgb_pixel(0, 255, 0), 2);
            }

            //save results in image form
            //std::string image_save_name = output_save_location + "test_save_image_" + version + num2str(idx, "_%03d.png");
            //save_png(rgb_img, image_save_name);

            test_results += tr;
            //dlib::sleep(50);
            //std::cin.ignore();

        }

        // output the test results
        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Testing Results (detction_accuracy, correct_hits, false_positives, missing_detections):  " << std::fixed << std::setprecision(4) << test_results(0, 0) / (double)test_file.size() << ", " << test_results(0, 3) << ", " << test_results(0, 4) << ", " << test_results(0, 5) << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;

        // save the results to the log file
        DataLogStream << "------------------------------------------------------------------" << std::endl;
        DataLogStream << "Training Results (detction_accuracy, correct_hits, false_positives, missing_detections): " << std::fixed << std::setprecision(4) << training_results(0, 0) / (double)training_file.size() << ", " << training_results(0, 3) << ", " << training_results(0, 4) << ", " << training_results(0, 5) << std::endl;
        DataLogStream << "Testing Results (detction_accuracy, correct_hits, false_positives, missing_detections):  " << std::fixed << std::setprecision(4) << test_results(0, 0) / (double)test_file.size() << ", " << test_results(0, 3) << ", " << test_results(0, 4) << ", " << test_results(0, 5) << std::endl;
        DataLogStream << "------------------------------------------------------------------" << std::endl;

        std::cout << "End of Program." << std::endl;
        DataLogStream.close();
        std::cin.ignore();
        
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;

        DataLogStream << std::endl;
        DataLogStream << "------------------------------------------------------------------" << std::endl;
        DataLogStream << e.what() << std::endl;
        DataLogStream << "------------------------------------------------------------------" << std::endl;
        DataLogStream.close();

        std::cout << "Press Enter to close..." << std::endl;
        std::cin.ignore();
    }

    return 0;

}	// end of main
