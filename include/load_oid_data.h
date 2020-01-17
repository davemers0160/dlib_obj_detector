#ifndef LOAD_OPEN_IMAGES_DATA_H
#define LOAD_OPEN_IMAGES_DATA_H

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <string>

// Custom Includes
#include "file_parser.h"

// dlib includes
#include <dlib/dnn.h>
#include <dlib/image_io.h>

// dlib-contrib includes
#include "rgb2gray.h"

extern const uint32_t array_depth;

// ----------------------------------------------------------------------------------------

void read_oid_labels(
    const std::vector<std::string> data_file,
    //std::vector<dlib::mmod_rect>& labels,
    dlib::mmod_rect& labels,
    uint64_t height,
    uint64_t width
)
{

    uint64_t left = (uint64_t)floor(std::stof(data_file[4]) * width);
    uint64_t right = (uint64_t)ceil(std::stof(data_file[5]) * width);
    uint64_t top = (uint64_t)floor(std::stof(data_file[6]) * height);
    uint64_t bottom = (uint64_t)ceil(std::stof(data_file[7]) * height);

    dlib::rectangle r(left, top, right, bottom);
    labels = dlib::mmod_rect(r, 0.0, data_file[2]);

    //labels.push_back(m_r);

}   // end of read_open_images_labels

// ----------------------------------------------------------------------------------------

template<typename img_type>
void load_oid_single_set(
    const std::string data_directory,
    const std::vector<std::string> data_file,
    img_type& img,
    //std::vector<dlib::mmod_rect>& labels
    dlib::mmod_rect& labels
)
{
    uint32_t start;
    long r, c;
    dlib::matrix<dlib::rgb_pixel> t1, t2;
    dlib::rgb_pixel p;

    std::string image_file = data_directory + data_file[0] + ".jpg";

    // load in the RGB image with 3 or more channels - ignoring everything after RGB		
    dlib::load_image(t1, image_file);

    for (uint32_t d = 0; d < array_depth; ++d)
    {
        img[d].set_size(t1.nr(), t1.nc());
    }

    switch (array_depth)
    {
    // case for converting an RGB image to grayscale image
    case 1:
        dlib::rgb2gray(t1, img[0]);
        start = 1;
        break;

    // case for using an single 3-channel image
    case 3:
        for (r = 0; r < t1.nr(); ++r)
        {
            for (c = 0; c < t1.nc(); ++c)
            {
                dlib::assign_pixel(p, t1(r, c));
                dlib::assign_pixel(img[0](r, c), p.red);
                dlib::assign_pixel(img[1](r, c), p.green);
                dlib::assign_pixel(img[2](r, c), p.blue);
            }
        }
        start = 1;
        break;

    }   // end of switch case

    // load in the label data
    read_oid_labels(data_file, labels, t1.nr(), t1.nc());

}   // end of load_single_set

// ----------------------------------------------------------------------------------------

template<typename img_type>
void load_oid_data(
    const std::vector<std::vector<std::string>> data_file,
    const std::string data_directory,
    std::vector<img_type>& img,
    std::vector<std::vector<dlib::mmod_rect>>& labels,
    std::vector<std::string>& image_files
)
{
    uint32_t idx;

    std::string image_file, depth_file;
    img_type t;

    img_type tmp_img;
    dlib::mmod_rect tmp_label;

    // clear out the container for the focus and defocus filenames
    img.clear();
    labels.clear();

    // load in the first image
    load_oid_single_set(data_directory, data_file[0], tmp_img, tmp_label);

    // push back the image and labels 
    img.push_back(tmp_img);
    labels.push_back({ tmp_label });
    image_files.push_back(data_directory + data_file[0][0] + ".jpg");

    if (data_file.size() < 2)
        return;

    std::vector<std::string>::iterator it;

    // run through the remaining images
    for (idx = 1; idx < data_file.size(); idx++)
    {
        //tmp_label.clear();

        // get the image filenames
        image_file = data_directory + data_file[idx][0] + ".jpg";

        // check the image to see if it has another set of boxes
        it = std::find(image_files.begin(), image_files.end(), image_file);
        
        if (it != image_files.end())
        {
            uint64_t index = std::distance(image_files.begin(), it);

            read_oid_labels(data_file[idx], tmp_label, img[index][0].nr(), img[index][0].nc());
            labels[index].push_back(tmp_label);
        }
        else
        {
            // load the image and labels
            load_oid_single_set(data_directory, data_file[idx], tmp_img, tmp_label); 
            
            // push back the image and labels
            image_files.push_back(image_file);
            img.push_back(tmp_img);
            labels.push_back({ tmp_label });
        }

    }   // end of the read in data loop

}   // end of load_oid_data

// ----------------------------------------------------------------------------------------

#endif  // LOAD_OPEN_IMAGES_DATA_H
