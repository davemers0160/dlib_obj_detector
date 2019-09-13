#ifndef LOAD_DFD_DATA_H
#define LOAD_DFD_DATA_H

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <string>

// Custom Includes
# include "file_parser.h"

// dlib includes
#include <dlib/dnn.h>
#include <dlib/image_io.h>
//#include <dlib/data_io.h>
//#include <dlib/image_transforms.h>


// extern const uint32_t array_depth;
// extern const uint32_t secondary;

// --------------------------------------------------------

template<typename img_type, typename pyramid_type, typename interpolation_type>
void dataset_downsample(
    img_type &img,
    std::vector<dlib::mmod_rect> &labels,
    const pyramid_type& pyr = dlib::pyramid_down<2>()
    //const interpolation_type& interp = dlib::interpolate_quadratic()
)
{
    uint64_t idx = 0;

    img_data_type tmp;      // this will store the intermediate results

    // 1. loop through the array layers
    //    - resize each layer according to pyr
    pyr(img);


    // 2. resize the label and reposition so that it bounds the right pixels
    //    - loop through each possible label and upsample
    //    - objects[i].rect = pyr.rect_up(objects[i].rect);
    for (idx = 0; idx < labels.size(); ++idx)
    {
        labels[idx].rect = pyr.rect_down(labels[idx].rect);
    }
}

// --------------------------------------------------------

void read_labels(
    const std::vector<std::string> data_file,
    std::vector<dlib::mmod_rect> &labels
)
{
    uint64_t idx = 0;

    // load in the label info
    for (idx = 2; idx < data_file.size(); idx += 5)
    {
        uint64_t left = std::stol(data_file[idx]);
        uint64_t top = std::stol(data_file[idx + 1]);
        uint64_t right = left + std::stol(data_file[idx + 2]);
        uint64_t bottom = top + std::stol(data_file[idx + 3]);
        dlib::rectangle r(left, top, right, bottom);
        dlib::mmod_rect m_r(r, 0.0, data_file[idx + 4]);

        labels.push_back(m_r);
    }

}   // end of read_labels

// --------------------------------------------------------

void read_group_labels(
    const std::vector<std::string> params,
    std::vector<dlib::mmod_rect> &labels
)
{
    uint64_t idx = 0, jdx = 0;
    uint64_t left, right, top, bottom;

    // load in the label info
    for (idx = 1; idx < params.size(); ++idx)
    {
        std::vector<std::string> label_info;

        parse_csv_line(params[idx], label_info);

        // get the label since it is the last element and the remove
        std::string label_name = label_info.back();
        label_info.pop_back();

        // convert the strings to uints
        std::vector<uint32_t> points(label_info.size());
        for (uint32_t jdx = 0; jdx < label_info.size(); ++jdx)
        {
            points[jdx] = (uint32_t)std::stoi(label_info[jdx]);
        }

        // check the size of points.  If there are more than 4 points then the input is
        // a polygon otherwise it is a rectangle
        if (points.size() < 4)
        {
            continue;
        }
        else if(points.size() == 4)
        {
            // create the rect from the x,y, w,h points
            left = points[0];
            top = points[1];
            right = left + points[2];
            bottom = top + points[3];
        }        
        else
        {
            // now assume that there are and equal number of x,y points
            uint32_t div = points.size() >> 1;

            const auto x = std::minmax_element(begin(points), begin(points) + div);
            const auto y = std::minmax_element(begin(points) + div, end(points));

            left = *x.first;
            right = *x.second;
            top = *y.first;
            bottom = *y.second;
        }


        dlib::rectangle r(left, top, right, bottom);
        dlib::mmod_rect m_r(r, 0.0, label_name);

        // add the new label to the list of labels
        labels.push_back(m_r);
    }

}   // end of read_group_labels


// --------------------------------------------------------

template<typename img_type>
void load_data(
    const std::vector<std::vector<std::string>> training_file, 
    const std::string data_directory,
    std::vector<img_type> &img,
	std::vector<std::vector<dlib::mmod_rect>> &labels,
    std::vector<std::string> &image_files
)
{

    uint32_t idx;

    std::string image_file, depth_file;

    // clear out the container for the focus and defocus filenames
	img.clear();
	labels.clear();

    for (idx = 0; idx < training_file.size(); idx++)
    {

        // get the image file
		image_file = data_directory + training_file[idx][0];
        image_files.push_back(image_file);
                
		// load in the RGB image with 3 or more channels - ignoring everything after RGB
		dlib::matrix<dlib::rgb_pixel> tmp_img;
        dlib::load_image(tmp_img, image_file);
		img.push_back(tmp_img);

		// load in the label info
		std::vector<dlib::mmod_rect> tmp_label;
        read_group_labels(training_file[idx], tmp_label);
        
		labels.push_back(tmp_label);

    }   // end of the read in data loop

}   // end of loadData


#endif  // LOAD_DFD_DATA_H