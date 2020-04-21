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

    }
    template<typename Dtype>
    void ConvolutionWeightLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top){
        CHECK(bottom[0]->shape()[1] == bottom[1]->shape()[1] && bottom[0]->shape()[2] == bottom[1]->shape()[2] && bottom[0]->shape()[3] == bottom[1]->shape()[3])
        << "Size Not Match";
        top[0]->Reshape({1, bottom[1]->shape()[2], 1, 1});
    }

    template<typename Dtype>
    void ConvolutionWeightLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                                    const vector<Blob<Dtype> *> &top) {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* bottom_data_conv = bottom[1]->cpu_data();

        Dtype* top_data = top[0]->mutable_cpu_data();
        const int count = top[0]->shape()[1];
        for (int i = 0; i < count; ++i) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, 1, bottom[0]->shape()[2],
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

