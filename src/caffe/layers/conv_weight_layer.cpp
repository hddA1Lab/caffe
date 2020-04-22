//
// Created by hdd on 2020/4/20.
//
#include <vector>

#include "caffe/layers/conv_weight_layer.hpp"
namespace caffe {
    template<typename Dtype>
    void ConvolutionWeightLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top){
        CHECK(bottom.size() == 2)
        << "no weight forwarding";

        ConvolutionWeightParameter cw_param = this->layer_param_.convolutionweight_param();
        kernel_channel_ = cw_param.kernel_c();
        kernel_height_ = cw_param.kernel_h();
        kernel_width_ = cw_param.kernel_w();
        kernel_num_ = cw_param.kernel_n();

//        kernel_channel_ = 512;
//        kernel_height_ = 4;
//        kernel_width_ = 4;
//        kernel_num_ = 20;
    }

    template<typename Dtype>
    void ConvolutionWeightLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top){
        const std::vector<int> bottom_shape = bottom[0]->shape();
        const std::vector<int> bottom_shape_conv = bottom[1]->shape();

        CHECK(bottom_shape[2] == kernel_width_ && bottom_shape[3] == kernel_height_ && bottom_shape[1] == kernel_channel_)
        << "Parameters Not Match";
        CHECK(bottom_shape[1] == bottom_shape_conv[1] && bottom_shape[2] == bottom_shape_conv[2] && bottom_shape[3] == bottom_shape_conv[3])
        << "Size Not Match";

        bottom_buffer_.Reshape({1, 1, kernel_width_ * kernel_height_ * kernel_channel_, 1});
        bottom_buffer_conv_.Reshape({kernel_num_, 1, 1, kernel_width_ * kernel_height_ * kernel_channel_});
        top[0]->Reshape({1, kernel_num_, 1, 1});

        conv_offset_ = bottom_buffer_.count();
    }

    template<typename Dtype>
    void ConvolutionWeightLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                                    const vector<Blob<Dtype> *> &top) {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* bottom_data_conv = bottom[1]->cpu_data();

        cwim2col_cpu(bottom_data, bottom_buffer_.mutable_cpu_data());
        cwim2col_cpu(bottom_data_conv, bottom_buffer_conv_.mutable_cpu_data());

        bottom_data = bottom_buffer_.cpu_data();
        bottom_data_conv = bottom_buffer_conv_.cpu_data();

        Dtype* top_data = top[0]->mutable_cpu_data();
        const int count = top[0]->count();

        for (int i = 0; i < count; ++i) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, 1, conv_offset_,
                                  (Dtype)1., &bottom_data_conv[i], bottom_data,
                                  (Dtype)0., &top_data[i]);
        }
    }

    template<typename Dtype>
    void ConvolutionWeightLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
        CHECK(false)
        << "Not Implemented";
    }

    template<typename Dtype>
    void ConvolutionWeightLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& top,
                                                    const vector<Blob<Dtype>*>& bottom){
        CHECK(false)
        << "Not Implemented";
    }

    template<typename Dtype>
    void ConvolutionWeightLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
        CHECK(false)
        << "Not Implemented";
    }


INSTANTIATE_CLASS(ConvolutionWeightLayer);
REGISTER_LAYER_CLASS(ConvolutionWeight);
}

