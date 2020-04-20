//
// Created by hdd on 2020/4/20.
//
#include <vector>

#include "caffe/layers/conv_weight_layer.hpp"
namespace caffe {
    template<typename Dtype>
    void ConvolutionWeightLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top){

    }
    template<typename Dtype>
    void ConvolutionWeightLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top){

    }

    template<typename Dtype>
    void ConvolutionWeightLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                                    const vector<Blob<Dtype> *> &top) {
        CHECK_EQ(bottom.size(), 2)
            << "no weight forwarding";

        this->top_dim_ = bottom[0]->count(0);
        Dtype* top_data = top[0]->mutable_cpu_data();

        for(int i = 0; i < this->top_dim_; i++){
            caffe_mul(1, &bottom[0]->cpu_data()[0], bottom[1]->cpu_data(), &top_data[0]);
        }
    }

    template<typename Dtype>
    void ConvolutionWeightLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){

    }

    template<typename Dtype>
    void ConvolutionWeightLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& top,
                                                    const vector<Blob<Dtype>*>& bottom){

    }

    template<typename Dtype>
    void ConvolutionWeightLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){

    }


INSTANTIATE_CLASS(ConvolutionWeightLayer);
REGISTER_LAYER_CLASS(ConvolutionWeight);
}

