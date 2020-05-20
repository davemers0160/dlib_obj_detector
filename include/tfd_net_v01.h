#ifndef NET_DEFINITION_H
#define NET_DEFINITION_H


// dlib includes
#include "dlib/dnn.h"
#include "dlib/dnn/core.h"

#include "input_array_image_pryamid.h"

extern const uint32_t array_depth = 1;
//extern const uint32_t secondary = 1;

// --------------------------------- Conv Filter Setup ------------------------------------
template <long num_filters, typename SUBNET> using con1 = dlib::con<num_filters, 1, 1, 1, 1, SUBNET>;
template <long num_filters, typename SUBNET> using con3 = dlib::con<num_filters, 3, 3, 1, 1, SUBNET>;
template <long num_filters, typename SUBNET> using con5 = dlib::con<num_filters, 5, 5, 1, 1, SUBNET>;
template <long num_filters, typename SUBNET> using con7 = dlib::con<num_filters, 7, 7, 1, 1, SUBNET>;
template <long num_filters, typename SUBNET> using con9 = dlib::con<num_filters, 9, 9, 1, 1, SUBNET>;

template <long num_filters, typename SUBNET> using con31 = dlib::con<num_filters, 3, 1, 1, 1, SUBNET>;
template <long num_filters, typename SUBNET> using con13 = dlib::con<num_filters, 1, 3, 1, 1, SUBNET>;
template <long num_filters, typename SUBNET> using con51 = dlib::con<num_filters, 5, 1, 1, 1, SUBNET>;
template <long num_filters, typename SUBNET> using con15 = dlib::con<num_filters, 1, 5, 1, 1, SUBNET>;
template <long num_filters, typename SUBNET> using con71 = dlib::con<num_filters, 7, 1, 1, 1, SUBNET>;
template <long num_filters, typename SUBNET> using con17 = dlib::con<num_filters, 1, 7, 1, 1, SUBNET>;
template <long num_filters, typename SUBNET> using con91 = dlib::con<num_filters, 9, 1, 1, 1, SUBNET>;
template <long num_filters, typename SUBNET> using con19 = dlib::con<num_filters, 1, 9, 1, 1, SUBNET>;


template <long num_filters, typename SUBNET> using con2d = dlib::con<num_filters, 2, 2, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con3d = dlib::con<num_filters, 3, 3, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con7d = dlib::con<num_filters, 7, 7, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con9d = dlib::con<num_filters, 9, 9, 2, 2, SUBNET>;

template <long num_filters, typename SUBNET> using cont2 = dlib::cont<num_filters, 2, 2, 1, 1, SUBNET>;
template <long num_filters, typename SUBNET> using cont3 = dlib::cont<num_filters, 3, 3, 1, 1, SUBNET>;
template <long num_filters, typename SUBNET> using cont5 = dlib::cont<num_filters, 5, 5, 1, 1, SUBNET>;
template <long num_filters, typename SUBNET> using cont7 = dlib::cont<num_filters, 7, 7, 1, 1, SUBNET>;
template <long num_filters, typename SUBNET> using cont9 = dlib::cont<num_filters, 9, 9, 1, 1, SUBNET>;

template <long num_filters, typename SUBNET> using cont2u = dlib::cont<num_filters, 2, 2, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using cont3u = dlib::cont<num_filters, 3, 3, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using cont4u = dlib::cont<num_filters, 4, 4, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using cont5u = dlib::cont<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using cont7u = dlib::cont<num_filters, 7, 7, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using cont9u = dlib::cont<num_filters, 9, 9, 2, 2, SUBNET>;

// ----------------------------------------------------------------------------------------

// --------------------------------------block definitions---------------------------------
template<typename SUBNET> using mp2 = dlib::max_pool<2, 2, 2, 2, SUBNET>;
template<typename SUBNET> using mp3 = dlib::max_pool<3, 3, 2, 2, SUBNET>;
template<typename SUBNET> using mp7 = dlib::max_pool<7, 7, 2, 2, SUBNET>;
template<typename SUBNET> using mp9 = dlib::max_pool<9, 9, 2, 2, SUBNET>;


template <int N1, int N2, int N3, typename SUBNET> using blk3 = dlib::bn_con<con1<N1, dlib::prelu<dlib::bn_con<con3<N2, dlib::prelu<dlib::bn_con<con1<N3, SUBNET>>>>>>>>;
template <int N1, int N2, int N3, typename SUBNET> using ablk3 = dlib::affine<con1<N1, dlib::prelu<dlib::affine<con3<N2, dlib::prelu<dlib::affine<con1<N3, SUBNET>>>>>>>>;

template <int N1, int N2, int N3, typename SUBNET> using blk5 = dlib::bn_con<con1<N1, dlib::prelu<dlib::bn_con<con5<N2, dlib::prelu<dlib::bn_con<con1<N3, SUBNET>>>>>>>>;
template <int N1, int N2, int N3, typename SUBNET> using ablk5 = dlib::affine<con1<N1, dlib::prelu<dlib::affine<con5<N2, dlib::prelu<dlib::affine<con1<N3, SUBNET>>>>>>>>;

template <int N, typename SUBNET> using cbp3_blk = dlib::prelu<dlib::bn_con<con3<N, SUBNET>>>;
template <int N, typename SUBNET> using cbp5_blk = dlib::prelu<dlib::bn_con<con5<N, SUBNET>>>;

template <int N, typename SUBNET> using acbp3_blk = dlib::prelu<dlib::affine<con3<N, SUBNET>>>;
template <int N, typename SUBNET> using acbp5_blk = dlib::prelu<dlib::affine<con5<N, SUBNET>>>;


// --------------------------------------residual blocks-----------------------------------

template <int N1, int N2, int N3, typename SUBNET>
using res_blk3 = dlib::prelu<dlib::add_prev1<blk3<N1, N2, N3, dlib::tag1<SUBNET>>>>;
template <int N1, int N2, int N3, typename SUBNET>
using ares_blk3 = dlib::prelu<dlib::add_prev1<ablk3<N1, N2, N3, dlib::tag1<SUBNET>>>>;

template <int N1, int N2, int N3, typename SUBNET>
using res_blk5 = dlib::prelu<dlib::add_prev1<blk5<N1, N2, N3, dlib::tag1<SUBNET>>>>;
template <int N1, int N2, int N3, typename SUBNET>
using ares_blk5 = dlib::prelu<dlib::add_prev1<ablk5<N1, N2, N3, dlib::tag1<SUBNET>>>>;



// Now we can define the 8x downsampling block in terms of conv5d blocks.  We
// also use relu and batch normalization in the standard way.
template <int N1, int N2, int N3, typename SUBNET> using downsampler = dlib::relu<dlib::bn_con<con5d<N1, dlib::relu<dlib::bn_con<con5d<N2, dlib::relu<dlib::bn_con<con5d<N3,SUBNET>>>>>>>>>;
template <int N1, int N2, int N3, typename SUBNET> using adownsampler = dlib::relu<dlib::affine<con5d<N1, dlib::relu<dlib::affine<con5d<N2, dlib::relu<dlib::affine<con5d<N3, SUBNET>>>>>>>>>;

// The rest of the network will be 3x3 conv layers with batch normalization and
// relu.  So we define the 3x3 block we will use here.
template <int N1, typename SUBNET> using rcon3 = dlib::relu<dlib::bn_con<con3<N1, SUBNET>>>;
template <int N1, typename SUBNET> using arcon3 = dlib::relu<dlib::affine<con3<N1, SUBNET>>>; 

template <int N1, typename SUBNET> using rcon5 = dlib::relu<dlib::bn_con<con5<N1,SUBNET>>>;
template <int N1, typename SUBNET> using arcon5 = dlib::relu<dlib::affine<con5<N1, SUBNET>>>;

/*

downsampler = in -> conv(5,5,2,2)[32] -> batch_normilization -> pReLU -> conv(5,5,2,2)[32] -> batch_normilization -> pReLU -> out

rcon3 = in -> conv(3,3,1,1)[32] -> batch_normilization -> pReLU -> out

input[4] -> downsampler -> rcon3 -> rcon3 -> rcon3 -> con6

*/

// ----------------------------------------------------------------------------------------

// Finally, we define the entire network.   The special input_rgb_image_pyramid
// layer causes the network to operate over a spatial pyramid, making the detector
// scale invariant.  

using net_type = dlib::loss_mmod<con7<1,

    dlib::relu<dlib::bn_con<con3<32,
    dlib::relu<dlib::bn_con<con3<32,
    dlib::relu<dlib::bn_con<con3<32,

    dlib::relu<dlib::bn_con<con5d<32, 
    dlib::relu<dlib::bn_con<con5d<32, 
    dlib::relu<dlib::bn_con<con5d<32,
    
    dlib::tag10<dlib::input_array_image_pyramid<dlib::pyramid_down<6>, array_depth>>
    >>> >>> >>> >>> >>> >>> >>;

using anet_type = dlib::loss_mmod<con7<1,

    dlib::relu<dlib::affine<con3<32,
    dlib::relu<dlib::affine<con3<32,
    dlib::relu<dlib::affine<con3<32,

    dlib::relu<dlib::affine<con5d<32, 
    dlib::relu<dlib::affine<con5d<32, 
    dlib::relu<dlib::affine<con5d<32,

    dlib::tag10<dlib::input_array_image_pyramid<dlib::pyramid_down<6>, array_depth>>
    >>> >>> >>> >>> >>> >>> >>;

// ----------------------------------------------------------------------------------------
// Configuration function
// ----------------------------------------------------------------------------------------

template <typename net_type>
net_type config_net(dlib::mmod_options options, std::vector<uint32_t> params)
{

    net_type net = net_type(options, dlib::num_con_outputs(params[0]),
        dlib::num_con_outputs(params[1]),
        dlib::num_con_outputs(params[2]),
        dlib::num_con_outputs(params[3]),
        dlib::num_con_outputs(params[4]),
        dlib::num_con_outputs(params[5]),
        dlib::num_con_outputs(params[6])
    );

    net.subnet().layer_details().set_num_filters(options.detector_windows.size());

    return net;

}   // end of config_net

// ----------------------------------------------------------------------------------------
//  GORGON Functions
// ----------------------------------------------------------------------------------------


//gorgon_capture<1> gc_01(256,128,3,3);
//gorgon_capture<5> gc_02(128,128,3,3);
//gorgon_capture<8> gc_03(128,192,3,3);
//gorgon_capture<14> gc_04(128,128,3,3);
//gorgon_capture<17> gc_05(128,320,3,3);
//gorgon_capture<23> gc_06(256,256,3,3);
//gorgon_capture<26> gc_07(256,576,3,3);
//gorgon_capture<32> gc_08(512,512,3,3);
//gorgon_capture<35> gc_09(512,64,3,3);
//gorgon_capture<40> gc_10(512,512,3,3);
//gorgon_capture<43> gc_11(512,64,3,3);
//gorgon_capture<48> gc_12(256,256,3,3);
//gorgon_capture<51> gc_13(256,64,3,3);
//gorgon_capture<56> gc_14(128,128,3,3);
//gorgon_capture<59> gc_15(128,img_depth,3,3);

void init_gorgon(std::string save_location)
{
    //gc_01.init((save_location + "l01"));
    //gc_02.init((save_location + "l05"));
    //gc_03.init((save_location + "l08"));
    //gc_04.init((save_location + "l14"));
    //gc_05.init((save_location + "l17"));
    //gc_06.init((save_location + "l23"));
    //gc_07.init((save_location + "l26"));
    //gc_08.init((save_location + "l32"));
    //gc_09.init((save_location + "l35"));
    //gc_10.init((save_location + "l40"));
    //gc_11.init((save_location + "l43"));
    //gc_12.init((save_location + "l48"));
    //gc_13.init((save_location + "l51"));
    //gc_14.init((save_location + "l56"));
    //gc_15.init((save_location + "l59"));

}

template<typename net_type>
void save_gorgon(net_type &dfd_net, uint64_t one_step_calls)
{
    //gc_01.save_params(dfd_net, one_step_calls);
    //gc_02.save_params(dfd_net, one_step_calls);
    //gc_03.save_params(dfd_net, one_step_calls);
    //gc_04.save_params(dfd_net, one_step_calls);
    //gc_05.save_params(dfd_net, one_step_calls);
    //gc_06.save_params(dfd_net, one_step_calls);
    //gc_07.save_params(dfd_net, one_step_calls);
    //gc_08.save_params(dfd_net, one_step_calls);
    //gc_09.save_params(dfd_net, one_step_calls);
    //gc_10.save_params(dfd_net, one_step_calls);
    //gc_11.save_params(dfd_net, one_step_calls);
    //gc_12.save_params(dfd_net, one_step_calls);
    //gc_13.save_params(dfd_net, one_step_calls);
    //gc_14.save_params(dfd_net, one_step_calls);
    //gc_15.save_params(dfd_net, one_step_calls);
}


void close_gorgon(void)
{
    //gc_01.close_stream();
    //gc_02.close_stream();
    //gc_03.close_stream();
    //gc_04.close_stream();
    //gc_05.close_stream();
    //gc_06.close_stream();
    //gc_07.close_stream();
    //gc_08.close_stream();
    //gc_09.close_stream();
    //gc_10.close_stream();
    //gc_11.close_stream();
    //gc_12.close_stream();
    //gc_13.close_stream();
    //gc_14.close_stream();
    //gc_15.close_stream();
}


#endif //NET_DEFINITION_H
