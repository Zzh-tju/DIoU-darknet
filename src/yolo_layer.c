#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

//#define DEBUG_NAN
//#define DEBUG_PRINTS

/**
 * total: total number of anchors
 */
layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes)
{
    int i;
    layer l = {0};
    l.type = YOLO;

    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = calloc(1, sizeof(float));
    // Anchor boxes
    l.biases = calloc(total*2, sizeof(float));
    // Which anchor boxes this layer is responsible for predicting
    if(mask) l.mask = mask;
    else{
        l.mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;
    l.truths = 90*(4 + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "yolo\n");
    srand(0);

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

/**
 * see yolov3.pdf, Section 2.1
 *   Tx,Ty,Tw,Th - predicted coordinates as offsets from anchor
 *   i/j == Cx/Cy - offset from top left corner of image
 * and https://github.com/pjreddie/darknet/issues/568#issuecomment-376291561
 *   x[...] - outputs of network
 *   biases[...] - anchors (bounding box prior)
 *   b.w, b.h - resulting width and height
 *
 * index: box index
 * (for each cell)
 *   i: col
 *   j: row
 * lw,lh: layer w,h
 * w,h: network w,h
 * stride: l.w*l.h
 *
 * returns:
 *   box in absolute coordinates
 */
box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride, REPRESENTATION representation)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
#ifdef DEBUG_NAN
    if (isnan(b.x)) {
      printf("get_yolo_box, isnan(b.x) is TRUE\n");
    }
    if (isnan(b.y)) {
      printf("get_yolo_box, isnan(b.y) is TRUE\n");
    }
#endif

    // With anchor boxes
    if (representation == REP_EXP) {
      // = Pw * e^(Tw)
      b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
      // = Ph * e^(Th)
      b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    }

    if (representation == REP_LIN) {
      if (x[index + 2*stride] <= 0) {
        //fprintf(stderr, "get_yolo_box w output is %f\n", x[index + 2*stride]);
        x[index + 2*stride] = exp(x[index + 2*stride]);
      }
      if (x[index + 3*stride] <= 0) {
        //fprintf(stderr, "get_yolo_box h output is %f\n", x[index + 3*stride]);
        x[index + 3*stride] = exp(x[index + 3*stride]);
      }
      // With anchor boxes, no exp
      // = Pw * Tw
      b.w = (x[index + 2*stride] * biases[2*n])   / w;
      // = Ph * Th
      b.h = (x[index + 3*stride] * biases[2*n+1]) / h;
    }

    return b;
}

/**
 * truth: ground truth bounding box
 * x: l.output
 * biases: anchors
 * n: highest iou index of anchor in l.total (total number of anchors)
 * index: box_index
 * stride: l.w * l.h
 * (for each)
 *   i: l.w
 *   j: l.h
 * // Layer w and h
 * lw: l.w
 * lh: l.h
 * // network w and h
 * w: net.w
 * h: net.h
 * scale: (2-truth.w*truth.h)
 * stride: l.w*l.h
 */
ious delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride, float iou_normalizer, IOU_LOSS iou_loss, REPRESENTATION representation)
{
    ious all_ious = {0};
    // i - step in layer width
    // j - step in layer height
    //  Returns a box in absolute coordinates
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride, representation);
    all_ious.iou = box_iou(pred, truth);
    all_ious.giou = box_giou(pred, truth);
    all_ious.diou = box_diou(pred, truth);
    all_ious.ciou = box_ciou(pred, truth);
    // avoid nan in dx_box_iou
    if (pred.w == 0) { pred.w = 1.0; }
    if (pred.h == 0) { pred.h = 1.0; }
    if (iou_loss == MSE) {
      float tx = (truth.x*lw - i);
      float ty = (truth.y*lh - j);
      float tw = log(truth.w*w / biases[2*n]);
      float th = log(truth.h*h / biases[2*n + 1]);

      delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
      delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
      delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
      delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    } else {
      all_ious.dx_iou = dx_box_iou(pred, truth, iou_loss);

      // jacobian^t (transpose)
      delta[index + 0*stride] = all_ious.dx_iou.dt;
      delta[index + 1*stride] = all_ious.dx_iou.db;
      delta[index + 2*stride] = all_ious.dx_iou.dl;
      delta[index + 3*stride] = all_ious.dx_iou.dr;

      // predict exponential, apply gradient of e^delta_t ONLY for w,h
      if (representation == REP_EXP) {
        delta[index + 2*stride] *= exp(x[index + 2*stride]);
        delta[index + 3*stride] *= exp(x[index + 3*stride]);
      }

      // normalize iou weight
      delta[index + 0*stride] *= iou_normalizer;
      delta[index + 1*stride] *= iou_normalizer;
      delta[index + 2*stride] *= iou_normalizer;
      delta[index + 3*stride] *= iou_normalizer;
    }

    //// DEBUG
#ifdef DEBUG_PRINTS
    printf("(x,y,w,h)\n p: (%f,%f,%f,%f)\n t: (%f,%f,%f,%f)\n", pred.x, pred.y, pred.w, pred.h, truth.x, truth.y, truth.w, truth.h);
    printf("delta [index: %d] %f", index, delta[index + 0*stride]);
    printf(",%f", delta[index + 1*stride]);
    printf(",%f", delta[index + 2*stride]);
    printf(",%f\n", delta[index + 3*stride]);
    printf("  delta: ");
    if ((pred.x < truth.x && delta[index + 0*stride] > 0) || (pred.x > truth.x && delta[index + 0*stride] < 0)) {
      printf("✓");
    } else {
      printf("𝒙");
    }
    printf(", ");
    if ((pred.y < truth.y && delta[index + 1*stride] > 0) || (pred.y > truth.y && delta[index + 1*stride] < 0)) {
      printf("✓");
    } else {
      printf("𝒙");
    }
    printf(", ");
    if ((pred.w < truth.w && delta[index + 2*stride] > 0) || (pred.w > truth.w && delta[index + 2*stride] < 0)) {
      printf("✓");
    } else {
      printf("𝒙");
    }
    printf(", ");
    if ((pred.h < truth.h && delta[index + 3*stride] > 0) || (pred.h > truth.h && delta[index + 3*stride] < 0)) {
      printf("✓");
    } else {
      printf("𝒙");
    }
    printf("\n");
#endif

    return all_ious;
}


/**
 * index: class_index
 */
void delta_yolo_class(float *output, float *delta, int index, int class, int classes, int stride, float *avg_cat)
{
    int n;
    if (delta[index]){
        delta[index + stride*class] = 1 - output[index + stride*class];
        printf("delta_yolo_class (cls %f)\n", delta[index + stride*class]);
        if(avg_cat) *avg_cat += output[index + stride*class];
        return;
    }
    for(n = 0; n < classes; ++n){
        delta[index + stride*n] = ((n == class)?1 : 0) - output[index + stride*n];
        if(n == class && avg_cat) *avg_cat += output[index + stride*n];
    }
}

static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

void forward_yolo_layer(const layer l, network net)
{
    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
    float tot_iou = 0;
    float tot_giou = 0;
    float tot_diou = 0;
    float tot_ciou = 0;
	
    float tot_iou_loss = 0;
    float tot_giou_loss = 0;
    float tot_diou_loss = 0;
    float tot_ciou_loss = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h, l.representation);
                    float best_iou = 0;
                    int best_t = 0;
                    for(t = 0; t < l.max_boxes; ++t){
                        box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    avg_anyobj += l.output[obj_index];
                    l.delta[obj_index] = l.cls_normalizer * (0 - l.output[obj_index]);
                    if (best_iou > l.ignore_thresh) {
                        l.delta[obj_index] = 0;
                    }
                    if (best_iou > l.truth_thresh) {
                        l.delta[obj_index] = l.cls_normalizer * (1 - l.output[obj_index]);

                        int class = net.truth[best_t*(4 + 1) + b*l.truths + 4];
                        if (l.map) class = l.map[class];
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0);
                        box truth = float_to_box(net.truth + best_t*(4 + 1) + b*l.truths, 1);
                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h, l.iou_normalizer, l.iou_loss, l.representation);
                    }
                }
            }
        }
        // For max boxes, given by max in cfg
        for(t = 0; t < l.max_boxes; ++t){
            box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            // for each anchor
            for(n = 0; n < l.total; ++n){
                box pred = {0};
                // resize the w/h w/ anchors
                pred.w = l.biases[2*n]/net.w;
                pred.h = l.biases[2*n+1]/net.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }

            int mask_n = int_index(l.mask, best_n, l.n);
            // object-ness
            if(mask_n >= 0){
                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                ious all_ious = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h, l.iou_normalizer, l.iou_loss, l.representation);
                // range is 0 <= 1
                tot_iou += all_ious.iou;
                tot_iou_loss += 1 - all_ious.iou;
                // range is -1 <= giou <= 1
                tot_giou += all_ious.giou;
                tot_giou_loss += 1 - all_ious.giou;
				
		tot_diou += all_ious.diou;
                tot_diou_loss += 1 - all_ious.diou;
				
		tot_ciou += all_ious.ciou;
                tot_ciou_loss += 1 - all_ious.ciou;
				
                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                avg_obj += l.output[obj_index];
                l.delta[obj_index] = l.cls_normalizer * (1 - l.output[obj_index]);

                int class = net.truth[t*(4 + 1) + b*l.truths + 4];
                if (l.map) class = l.map[class];
                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, &avg_cat);

                ++count;
                ++class_count;
                if(all_ious.iou > .5) recall += 1;
                if(all_ious.iou > .75) recall75 += 1;
            }
        }
    }
    // Always compute classification loss both for iou + cls loss and for logging with mse loss
    // TODO: remove IOU loss fields before computing MSE on class
    //   probably split into two arrays
    int stride = l.w*l.h;
    float* no_iou_loss_delta = calloc(l.batch * l.outputs, sizeof(float));
    memcpy(no_iou_loss_delta, l.delta, l.batch * l.outputs * sizeof(float));
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    no_iou_loss_delta[index + 0*stride] = 0;
                    no_iou_loss_delta[index + 1*stride] = 0;
                    no_iou_loss_delta[index + 2*stride] = 0;
                    no_iou_loss_delta[index + 3*stride] = 0;
                }
            }
        }
    }
    float classification_loss = l.cls_normalizer * pow(mag_array(no_iou_loss_delta, l.outputs * l.batch), 2);
    free(no_iou_loss_delta);

    float avg_iou_loss = 0;
    // gIOU loss + MSE (objectness) loss
    if (l.iou_loss == MSE ) {
      *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    } else {
      if (l.iou_loss == IOU) {
        avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_iou_loss / count) : 0;
      } else {
            if (l.iou_loss == GIOU) {
              avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_giou_loss / count) : 0;
            } else {
                if (l.iou_loss == DIOU) {
                  avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_diou_loss / count) : 0;
                } else {
                    avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_ciou_loss / count) : 0;
                    } 
              }
        }
      *(l.cost) = avg_iou_loss + classification_loss;
    }
//#ifdef DEBUG_PRINTS
//    printf("iou_loss: %f, iou_loss_count: %d, avg_iou_loss: %f, classification_loss: %f, total_cost: %f\n", iou_loss, count, avg_iou_loss, classification_loss, *(l.cost));
//#endif
    printf("v3 (%s loss, Normalizer: (iou: %f, cls: %f) Region %d Avg (IOU: %f, %s: %f), Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f, count: %d\n", (l.iou_loss==MSE?"mse":(l.iou_loss==IOU?"iou":(l.iou_loss==GIOU?"giou":(l.iou_loss==DIOU?"diou":"ciou")))), l.iou_normalizer, l.cls_normalizer, net.index, tot_iou/count, (l.iou_loss==GIOU?"GIOU":(l.iou_loss==DIOU?"DIOU":"CIOU")), (l.iou_loss==GIOU? tot_giou/count:(l.iou_loss==DIOU? tot_diou/count:tot_ciou/count)), avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);
    if (count > 0 && strlen(net.logfile) != 0) {
      log_avg_iou(net.logfile, tot_iou, tot_giou, count, avg_iou_loss, classification_loss, *(l.cost));
    }
}

void backward_yolo_layer(const layer l, network net)
{
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

/**
 * Converts output of the network to detection boxes
 * w,h: image width,height
 * netw,neth: network width,height
 * relative: 1 (all callers seems to pass TRUE)
 */
void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    // network height (or width)
    int new_w=0;
    // network height (or width)
    int new_h=0;
    // Compute scale given image w,h vs network w,h
    // I think this "rotates" the image to match network to input image w/h ratio
    // new_h and new_w are really just network width and height
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    // difference between network width and "rotated" width
    float deltaw = netw - new_w;
    // difference between network height and "rotated" height
    float deltah = neth - new_h;
    // ratio between rotated network width and network width
    float ratiow = (float)new_w/netw;
    // ratio between rotated network width and network width
    float ratioh = (float)new_h/neth;
    for (i = 0; i < n; ++i){

        box b = dets[i].bbox;
        // x = ( x - (deltaw/2)/netw ) / ratiow;
        //   x - [(1/2 the difference of the network width and rotated width) / (network width)]
        b.x =  (b.x - deltaw/2./netw) / ratiow;
        b.y =  (b.y - deltah/2./neth) / ratioh;
        // scale to match rotation of incoming image
        b.w *= 1/ratiow;
        b.h *= 1/ratioh;

        // relative seems to always be == 1, I don't think we hit this condition, ever.
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }

        dets[i].bbox = b;
    }
}

int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}

/**
 * Converts output of the network to detection boxes
 * w,h: image width,height
 * netw,neth: network width,height
 */
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;
    if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h, l.representation);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j = 0; j < l.classes; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_gpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_yolo_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_yolo_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

