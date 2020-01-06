#ifndef _EVAL_NET_PERFORMANCE_
#define _EVAL_NET_PERFORMANCE_

// This loading function assumes that the ground truth image size and the input image sizes do not have to be the same dimensions

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <string>

// Custom Includes
//#include "center_cropper.h"
//#include "dlib_matrix_threshold.h"
//#include "cyclic_analysis.h"

// dlib includes
#include <dlib/dnn.h>
#include <dlib/matrix.h>
#include <dlib/image_io.h>
#include <dlib/data_io.h>
#include <dlib/image_transforms.h>
#include <dlib/dnn/validation.h>

extern const uint32_t array_depth;


//-----------------------------------------------------------------------------

class fda_test_box_overlap
{
public:
    fda_test_box_overlap() : iou_thresh(0.5), percent_covered_thresh(1.0)
    {}

    explicit fda_test_box_overlap(
        double iou_thresh_,
        double percent_covered_thresh_ = 1.0
    ) : iou_thresh(iou_thresh_), percent_covered_thresh(percent_covered_thresh_)
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(0 <= iou_thresh && iou_thresh <= 1 &&
            0 <= percent_covered_thresh && percent_covered_thresh <= 1,
            "\t fda_test_box_overlap(iou_thresh, percent_covered_thresh)"
            << "\n\t Invalid inputs were given to this function "
            << "\n\t iou_thresh:   " << iou_thresh
            << "\n\t percent_covered_thresh: " << percent_covered_thresh
            << "\n\t this: " << this
        );

    }

    bool operator() (
        const dlib::rectangle& a,
        const dlib::rectangle& b
        ) const
    {
        const double inner = a.intersect(b).area();
        if (inner == 0)
            return false;

        //const double outer = (a+b).area();
        const double outer = a.area() + b.area() - inner;

        if (inner / outer > iou_thresh ||
            inner / a.area() > percent_covered_thresh ||
            inner / b.area() > percent_covered_thresh)
            return true;
        else
            return false;
    }

    double get_percent_covered_thresh(
    ) const
    {
        return percent_covered_thresh;
    }

    double get_iou_thresh(
    ) const
    {
        return iou_thresh;
    }

private:
    double iou_thresh;
    double percent_covered_thresh;

};	// end of class fda_test_box_overlap

//-----------------------------------------------------------------------------

inline bool fda_overlaps_any_box(
    const fda_test_box_overlap &tester,
    const std::vector<dlib::rectangle> &rects,
    const dlib::rectangle &rect
)
{
    for (uint64_t idx = 0; idx < rects.size(); ++idx)
    {
        if (tester(rects[idx], rect))
            return true;
    }
    return false;
}

//-----------------------------------------------------------------------------

inline uint64_t get_number_of_truth_hits(
    //const std::vector<dlib::full_object_detection> &ground_truth_boxes,
    const std::vector<dlib::mmod_rect> &ground_truth_boxes,
    const std::vector<dlib::rectangle> &ignore,
    //const std::vector<std::pair<double, dlib::rectangle>> &boxes,
    const std::vector<dlib::mmod_rect> &boxes,
    const fda_test_box_overlap &overlap_tester,
    std::vector<std::pair<double, bool>> &all_dets,
    uint64_t &missing_detections
)
/*!
    ensures
        - returns the number of elements in ground_truth_boxes which are overlapped by an
          element of boxes.  In this context, two boxes, A and B, overlap if and only if
          overlap_tester(A,B) == true.
        - No element of boxes is allowed to account for more than one element of ground_truth_boxes.
        - The returned number is in the range [0,ground_truth_boxes.size()]
        - Adds the score for each box from boxes into all_dets and labels each with
          a bool indicating if it hit a truth box.  Note that we skip boxes that
          don't hit any truth boxes and match an ignore box.
        - Adds the number of truth boxes which didn't have any hits into
          missing_detections.
!*/
{
    if (boxes.size() == 0)
    {
        missing_detections += ground_truth_boxes.size();
        return 0;
    }
    uint64_t idx, jdx;

    uint64_t count = 0;
    std::vector<bool> used(boxes.size(), false);

    for (idx = 0; idx < ground_truth_boxes.size(); ++idx)
    {
        bool found_match = false;
        // Find the first box that hits ground_truth_boxes[idx]
        for (jdx = 0; jdx < boxes.size(); ++jdx)
        {
            if (used[jdx])
                continue;

            //if (overlap_tester(ground_truth_boxes[idx].get_rect(), boxes[jdx].second))
            if ((overlap_tester(ground_truth_boxes[idx].rect, boxes[jdx].rect)) && (ground_truth_boxes[idx].label.compare(boxes[jdx].label) == 0))
            {
                used[jdx] = true;
                ++count;
                found_match = true;
                break;
            }
        }

        if (!found_match)
            ++missing_detections;
    }

    //for (idx = 0; idx < boxes.size(); ++idx)
    //{
    //    // only out put boxes if they match a truth box or are not ignored.
    //    if (used[idx] || !fda_overlaps_any_box(overlap_tester, ignore, boxes[idx].rect))
    //    {
    //        all_dets.push_back(std::make_pair(boxes[idx].detection_confidence, used[idx]));
    //    }
    //}

    return count;
}



//-----------------------------------------------------------------------------

// this function will perform the evaluation of the network performance for an
// img_type of dlib::matrix<dlib::rgb_pixel> 
template <typename net_type, typename img_type>

//to use the original net for evaluation, put an & before 'net' in net_type; to duplicate net, remove &
dlib::matrix<double, 1, 6> eval_net_performance(
    net_type &net,
    img_type &input_img,    
    std::vector<dlib::mmod_rect> &gt_labels,
    std::vector<dlib::mmod_rect> &dnn_labels,
    uint32_t min_target_size,
    fda_test_box_overlap overlap_tester
)
{
    uint64_t idx = 0;
    uint64_t missing_detections = 0;
    uint64_t small_missing_detections = 0;
    uint64_t false_positives = 0;
    uint64_t correct_hits = 0;
    uint64_t small_correct_hits = 0;
    uint64_t num_gt = 0;
    uint64_t num_dets = 0;

    double detction_accuracy = 0.0; 

    std::vector<std::pair<double, bool>> all_detections;
    std::vector<dlib::rectangle> ignore_boxes;
    //std::vector<std::pair<double, dlib::rectangle>> boxes;
    //std::vector<dlib::full_object_detection> truth_boxes;
    //std::vector<dlib::full_object_detection> small_truth_boxes;

    //std::vector<dlib::mmod_rect> boxes;
    std::vector<dlib::mmod_rect> truth_boxes;
    std::vector<dlib::mmod_rect> small_truth_boxes;

    // generate the cropper to slice the image if it is too big
    //cyclic_analysis ca_cropper;
    //ca_cropper.set_chip_dims(640, 640);
    //ca_cropper.set_overlap(60, 60);
    //ca_cropper.set_overlap_threshold(0.4);
    //
    //std::vector<dlib::mmod_rect> dets;
    // check the image size and see if it is too large.  If so 
  //  if (input_img[0].size() > (800 * 800))
  //  {
		////if 
  //      // run the input image through the cyclic analysis cropper
  //      dets = ca_cropper(net, input_img);
  //  }
  //  else
  //  {
  //      // run the input image through the network to get the detections
        //dets = net(input_img);
        dnn_labels = net(input_img);
  //  }

    // cycle through the ground truth labels and put them in the right bins (truth box, ignore)
    for (idx = 0; idx < gt_labels.size(); ++idx)
    {
        // copy truth_dets into the correct object
        if (gt_labels[idx].ignore)
        {
            ignore_boxes.push_back(gt_labels[idx].rect);
        }
        //else if ((gt_labels[idx].rect.width() < min_target_size) || (gt_labels[idx].rect.height() < min_target_size))
        //{
        //    small_truth_boxes.push_back(dlib::full_object_detection(gt_labels[idx].rect));
        //}
        else
        {
            //truth_boxes.push_back(dlib::full_object_detection(gt_labels[idx].rect));
            truth_boxes.push_back(gt_labels[idx]);
        }
    }

    // cycle through the detections 
/*    for (idx = 0; idx < dets.size(); ++idx)
    {
        boxes.push_back(std::make_pair(dets[idx].detection_confidence, dets[idx].rect));
        dnn_labels.push_back(dets[idx]);
    }   */ 

    // get the correct number of detections
    correct_hits = get_number_of_truth_hits(truth_boxes, ignore_boxes, dnn_labels, overlap_tester, all_detections, missing_detections);
    // check the boxes that are smaller than should be detected
    //small_correct_hits = get_number_of_truth_hits(small_truth_boxes, ignore, boxes, overlap_tester, all_detections, small_missing_detections);

    correct_hits += small_correct_hits;
    num_gt = truth_boxes.size() + small_correct_hits;       // number of ground truth images that are not ignored
    //num_dets = dets.size();					                // number of detections
    num_dets = dnn_labels.size();					                // number of detections

    false_positives = num_dets - correct_hits;

    if ((num_gt != 0) || (num_dets != 0))
    {
        detction_accuracy = (double)correct_hits / ((double)(num_gt + num_dets) / 2.0);
    }

    dlib::matrix<double, 1, 6> res;
    res = detction_accuracy, (double)num_gt, (double)num_dets, (double)correct_hits, (double)false_positives, (double)missing_detections;
    return res;

}   // end of eval_net_performance_matrix

//-----------------------------------------------------------------------------

//template <typename net_type, typename img_type>
//dlib::matrix<double, 1, 6> eval_net_performance(
//    net_type &net,
//    img_type &input_img,    
//    std::vector<dlib::mmod_rect> &gt_labels,
//    std::vector<dlib::mmod_rect> &dnn_labels
//)
//{
//    dlib::matrix<double, 1, 6> results = dlib::zeros_matrix<double>(1,6);
//    const fda_test_box_overlap overlap_tester = fda_test_box_overlap(0.3, 1.0);
//    dnn_labels.clear();
//
//    // check to see if the input image is a dlib::matrix object
//    // if not then we can assume that the img_type is an array of dlib::matrix objects
//    //if (dlib::is_matrix<img_type>::value == true)
//    //{
//        results = eval_net_performance_matrix(net, input_img, gt_labels, dnn_labels, overlap_tester);
//    //}
//    //else
//    //{
//    //    results = eval_net_performance_array(net, input_img, gt_labels, dnn_labels, overlap_tester);
//    //}
//    
//    return results;
//
//} // end of eval_all_net_performance

//-----------------------------------------------------------------------------

template <typename net_type, typename img_type>
dlib::matrix<double, 1, 6> eval_all_net_performance(
    net_type &net,
    std::vector<img_type> &input_img,    
    std::vector<std::vector<dlib::mmod_rect>> &gt_labels,
    std::vector<std::vector<dlib::mmod_rect>> &dnn_labels,
    uint32_t min_target_size
)
{
    uint64_t idx, jdx;
    std::vector<dlib::mmod_rect> dnn_lab;

    // this checks to make sure that there are the same number of images as labels
    DLIB_CASSERT(input_img.size() == gt_labels.size(), "The number of input images does not match the number of input labels");
    
    dnn_labels.clear();
	//net.clean();

    dlib::matrix<double, 1, 6> results = dlib::zeros_matrix<double>(1,6);
    const fda_test_box_overlap overlap_tester = fda_test_box_overlap(0.3, 1.0);

    // loop through the input images and get the results for each
	for (idx = 0; idx < input_img.size(); ++idx)
	{
		bool lbl = false;
		dnn_lab.clear();		// clear out the re-used container in each loop
		for (jdx = 0; jdx < gt_labels[idx].size(); jdx++) {
			lbl |= gt_labels[idx][jdx].ignore;
		}

		if(lbl == true){


			//dlib::pyramid_up(input_img[idx][0]);
			//dlib::pyramid_up(input_img[idx][1]);
			//dlib::pyramid_up(input_img[idx][2]);
			//
			////cycle through each label to scale left right top bottom
			//for(jdx = 0; jdx<gt_labels[idx].size(); jdx++){
			//	gt_labels[idx][jdx].rect = scale_rect(gt_labels[idx][jdx].rect, 2);
			//	gt_labels[idx][jdx].ignore = false;
			//}
			//

			//cycle through each one of the layers in the array and upscale them individually
			results += eval_net_performance(net, input_img[idx], gt_labels[idx], dnn_lab, min_target_size, overlap_tester);
			dnn_labels.push_back(dnn_lab);
		}
		else {
			results += eval_net_performance(net, input_img[idx], gt_labels[idx], dnn_lab, min_target_size, overlap_tester);
			dnn_labels.push_back(dnn_lab);
		}

    }
    
	results(0, 0) = results(0, 0) / (double)(input_img.size());

    // return the averaged results across all of the input images
    //return (results / (double)(input_img.size()));
	return results;

} // end of eval_all_net_performance

//-----------------------------------------------------------------------------


#endif  //_EVAL_NET_PERFORMANCE_