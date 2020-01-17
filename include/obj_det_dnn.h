#ifndef DFD_DNN_H_
#define DFD_DNN_H_


#include <cstdint>
#include <tuple>
//#include <utility>

// custom includes
#include "file_parser.h"

// dlib includes
//#include <dlib/dnn.h>

extern const uint32_t array_depth;
// extern const uint32_t secondary;

// ----------------------------------------------------------------------------------------

typedef struct training_params {

    training_params() = default;
    training_params(double ilr, double flr, double lrsf, uint32_t step) : intial_learning_rate(ilr), final_learning_rate(flr), learning_rate_shrink_factor(lrsf), steps_wo_progess(step){}

    double intial_learning_rate;
    double final_learning_rate;
    double learning_rate_shrink_factor;
    uint32_t steps_wo_progess;

} training_params;

// ----------------------------------------------------------------------------------------

typedef struct crop_info {

    crop_info() = default;
    crop_info(uint64_t n, uint64_t h, uint64_t w, double a) : crop_num(n), crop_height(h), crop_width(w), angle(a) {}

    uint64_t crop_num;
    uint64_t crop_height;
    uint64_t crop_width;
    double angle;

} crop_info;


// ----------------------------------------------------------------------------------------
void parse_input_file(std::string parse_filename, 
    std::string &version, 
    std::vector<int32_t> &gpu, 
    std::vector<double> &stop_criteria, 
    training_params &tp, 
    //std::string &training_file, 
    //std::string &test_file, 
    std::pair<std::string, uint8_t> &train_input,
    std::pair<std::string, uint8_t> &test_input,
    crop_info &ci, 
    std::pair<uint32_t, uint32_t> &target_size, 
    std::vector<uint32_t> &filter_num,
    std::string &save_directory
)
{


    std::vector<std::vector<std::string>> params;
    parse_csv_file(parse_filename, params);

    for (uint64_t idx = 0; idx<params.size(); ++idx)
    {
        switch (idx)
        {

            // get the version name of the network - used for naming various files
            case 0:
                version = params[idx][0];
                break;

            // select which gpu to use
		    case 1:
                try {
                    gpu.clear();
                    for (int jdx = 0; jdx < params[idx].size(); ++jdx)
                    {
                        gpu.push_back(stol(params[idx][jdx]));
                    }
                }
                catch (std::exception &e) {
                    std::cout << e.what() << std::endl;
                    gpu.clear();
                    gpu.push_back(0);
                }
			    break;

            // get the stopping criteria: max hours, max training steps
            case 2:
                try {

                    stop_criteria.clear();
                    for (uint64_t jdx = 0; jdx<params[idx].size(); ++jdx)
                    {
                        stop_criteria.push_back(stod(params[idx][jdx]));
                    }
                }
                catch (std::exception &e) {
                    std::cout << e.what() << std::endl;
                    stop_criteria.push_back(160.0);
                    stop_criteria.push_back(250000.0);
                    std::cout << "Error getting stopping criteria.  Setting values to default." << std::endl;
                }
                break;

            // get the training parameters
            case 3:
                try {
                    tp = training_params(stod(params[idx][0]), stod(params[idx][1]), stod(params[idx][2]), stol(params[idx][3]));
                }
                catch (std::exception &e) {
                    std::cout << e.what() << std::endl;
                    std::cout << "Using default training parameters..." << std::endl;
                    tp = training_params(0.001, 0.000001, 0.1, 2500);
                }
                break;

            // get the file that contains the training data
            case 4:
                try {
                    train_input = std::make_pair(params[idx][0], std::stoi(params[idx][1]));
                }
                catch (std::exception & e) {
                    std::cout << e.what() << std::endl;
                }
                break;

            // get the file that contains the test data
            case 5:
                try {
                    test_input = std::make_pair(params[idx][0], std::stoi(params[idx][1]));
                }
                catch (std::exception & e) {
                    std::cout << e.what() << std::endl;
                }
                break;

            // get the number of crops used for training
            case 6:
                try {
                    ci = crop_info(stol(params[idx][0]), stol(params[idx][1]), stol(params[idx][2]), stod(params[idx][3]));
                }
                catch (std::exception &e) {
                    std::cout << e.what() << std::endl;
                    std::cout << "Setting crop-info to defalut values..." << std::endl;
                    ci = crop_info(4, 200, 200, 10.0);
                }
                break;

            //// get the size of the crops (rows, cols) and the angle rotations (max, step)
            //case 7:
            //    try {
            //        if (params[idx].size() == 3)
            //        {
            //            crop_info = std::make_tuple(stol(params[idx][0]), stol(params[idx][1]), stod(params[idx][2]));
            //            //crop_size = std::make_pair(stol(params[idx][0]), stol(params[idx][1]));
            //        }
            //        //else if (params[idx].size() == 4)
            //        else
            //        {
            //            crop_info = std::make_tuple(300, 300, 15.0);
            //            //crop_size = std::make_pair(stol(params[idx][0]), stol(params[idx][1]));
            //            //angles = std::make_pair(stod(params[idx][2]), stod(params[idx][3]));
            //        }
            //    }
            //    catch (std::exception &e) {
            //        std::cout << e.what() << std::endl;
            //        //crop_size = std::make_pair(200, 200);
            //        crop_info = std::make_tuple(300, 300, 15.0);
            //        //angles = std::make_pair(15.0, 15.0);
            //        std::cout << "Setting crop size to (row, col, angle): " << std::get<0>(crop_info) << ", " << std::get<1>(crop_info) << ", " << std::get<2>(crop_info) << std::endl;
            //        //std::cout << "Setting angles to (max, step): " << angles.first << "," << angles.second << std::endl;
            //    }
            //    break;

            // get the min/max target sizes
            case 7:
                try {
                    target_size = std::make_pair(stol(params[idx][0]), stol(params[idx][1]));
                }
                catch (std::exception &e) {
                    std::cout << e.what() << std::endl;
                    target_size = std::make_pair(35, 70);
                }
                break;

            // get the number conv filters for each layer
            case 8:
                try {
                    filter_num.clear();
                    for (int jdx = 0; jdx<params[idx].size(); ++jdx)
                    {
                        filter_num.push_back(stol(params[idx][jdx]));
                    }
                }
                catch (std::exception &e) {
                    std::cout << e.what() << std::endl;
                    filter_num.clear();

                    std::cout << "Error getting filter numbers.  No values passed on." << std::endl;
                }
                break;

            case 9:
                save_directory = params[idx][0];
                break;

            default:
                break;
        }   // end of switch

    }   // end of for

}   // end of parse_dnn_data_file

#endif  // OBJ_DET_DNN_H_
