//
// Created by hdd on 2020/4/20.
//

#ifndef CAFFE_CONV_WEIGHT_LAYER_HPP
#define CAFFE_CONV_WEIGHT_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    template <typename Dtype>
    class ConvolutionWeightLayer : public Layer<Dtype> {
    public:
        explicit ConvolutionWeightLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}

        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "ConvolutionWeight"; }
        virtual inline int MinBottomBlobs() const { return 2; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

        Blob<Dtype> bottom_buffer_;
        Blob<Dtype> bottom_buffer_conv_;

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        int kernel_width_;
        int kernel_height_;
        int kernel_channel_;
        int kernel_num_;

        int conv_offset_;

    private:
        inline void cwim2col_cpu(const Dtype* data, Dtype* col_buff) {
            im2col_cpu(data,
                    kernel_channel_,
                    kernel_height_, kernel_width_,
                    kernel_height_, kernel_width_,
                    0, 0, 1, 1, 1, 1, col_buff);
        }

    };
}  // namespace caffe

#endif //CAFFE_CONV_WEIGHT_LAYER_HPP
