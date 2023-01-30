#!/usr/bin/env python
'''
Keisuke Suzuki <ksk@chain.hokudai.ac.jp>
2023-01-27
'''
import os
os.environ['GLOG_minloglevel'] = '2'  # suprress Caffe verbose prints

import settings
import site
site.addsitedir(settings.caffe_root)
import caffe

import numpy as np
import math, random
import sys, subprocess
from IPython.display import clear_output, Image, display
from scipy.misc import imresize
from numpy.linalg import norm
from numpy.testing import assert_array_equal
import scipy.misc, scipy.io
import patchShow
import argparse # parsing arguments
import scipy.ndimage


mean = np.float32([104.0, 117.0, 123.0])

fc_layers = ["fc6", "fc7", "fc8", "loss3/classifier", "fc1000", "prob"]
conv_layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]

if settings.gpu:
  caffe.set_mode_gpu() # uncomment this if gpu processing is available


def get_code(path, layer):
  '''
  Push the given image through an encoder to get a code.
  '''

  # set up the inputs for the net: 
  batch_size = 1
  image_size = (3, 227, 227)
  images = np.zeros((batch_size,) + image_size, dtype='float32')

  in_image = scipy.misc.imread(path)
  in_image = scipy.misc.imresize(in_image, (image_size[1], image_size[2]))

  for ni in range(images.shape[0]):
    images[ni] = np.transpose(in_image, (2, 0, 1))

  # Convert from RGB to BGR
  data = images[:,::-1] 

  # subtract the ImageNet mean
  matfile = scipy.io.loadmat('ilsvrc_2012_mean.mat')
  image_mean = matfile['image_mean']
  topleft = ((image_mean.shape[0] - image_size[1])/2, (image_mean.shape[1] - image_size[2])/2)
  image_mean = image_mean[int(topleft[0]):int(topleft[0]+image_size[1]), int(topleft[1]):int(topleft[1]+image_size[2])]
  del matfile
  data -= np.expand_dims(np.transpose(image_mean, (2,0,1)), 0) # mean is already BGR

  # initialize the encoder
  encoder = caffe.Net(settings.encoder_definition, settings.encoder_weights, caffe.TEST)

  # run encoder and extract the features
  encoder.forward(data=data)
  feat = np.copy(encoder.blobs[layer].data)
  del encoder

  zero_feat = feat[0].copy()[np.newaxis]

  return zero_feat, data


def make_step_generator(net, x, x0, start, end, step_size=1):
  '''
  Forward and backward passes through the generator DNN.
  '''

  src = net.blobs[start] # input image is stored in Net's 'data' blob
  dst = net.blobs[end]

  # L2 distance between init and target vector
  net.blobs[end].diff[...] = (x-x0)
  net.backward(start=end)
  g = net.blobs[start].diff.copy()

  grad_norm = norm(g)

  # reset objective after each step
  dst.diff.fill(0.)

  # If norm is Nan, skip updating the image
  if math.isnan(grad_norm):
    return 1e-12, src.data[:].copy()  
  elif grad_norm == 0:
    return 0, src.data[:].copy()

  # Make an update
  src.data[:] += step_size/np.abs(g).mean() * g

  return grad_norm, src.data[:].copy()


def make_step_net(net, end, unit, image, xy=0, step_size=1, act_mode="fixed", jitter=0, rotation=0, stats=False, stats_step=[], actmax_step=[]):
  '''
  Forward and backward passes through the DNN being visualized.
  '''

  src = net.blobs['data'] # input image
  dst = net.blobs[end]

  #jitter from deep dream
  ox, oy = np.random.randint(-jitter, jitter+1, 2)
  image = np.roll(np.roll(image, ox, -1), oy, 2)

  #rotation
  if not rotation == 0:
    orot = np.random.randint(-rotation, rotation + 1)
    tmp_img = np.squeeze(image, axis=0)
    tmp_img = tmp_img.transpose(1, 2, 0)
    tmp_img= scipy.ndimage.rotate(tmp_img, orot, reshape=False, mode='mirror')
    tmp_img = tmp_img.transpose(2, 0, 1)
    image = np.expand_dims(tmp_img, 0)
    
  acts = net.forward(data=image, end=end)

  if stats:
    if end in fc_layers:
      fc = acts[end][0]    
    elif end in conv_layers:
      #fc = acts[end][0, :, xy, xy]
      fc = acts[end][0, :, :, :]
      
      #fc = fc.max(axis=2).max(axis=1)
      fc = fc.mean(axis=2).mean(axis=1)
      #print(fc.shape)
      
    ## list the 10 maximum activations
    indices  = (-fc).argsort()[:10]
    for i in range(10):
      actmax_step.append(indices[i])
      actmax_step.append(fc[indices[i]])
      
    for act in fc:
      stats_step.append(act)
        
    
  one_hot = np.zeros_like(dst.data)
  if act_mode == "fixed":
  
    # Move in the direction of increasing activation of the given neuron
    if end in fc_layers:
      one_hot.flat[unit] = 1.
    elif end in conv_layers:
      #one_hot[:, unit, xy, xy] = 1. 
      one_hot[:, unit, :, :] = 1. # targeting layers for all space
    else:
      raise Exception("Invalid layer type!")
  
    dst.diff[:] = one_hot
  elif act_mode == "l2norm":
    # L2 norm
    dst.diff[:] = dst.data
    
  elif act_mode == "winner":

    if end in fc_layers:
      fc = acts[end][0]
      best_unit = fc.argmax()      
      one_hot.flat[best_unit] = 1.
    elif end in conv_layers:
      fc = acts[end][0, :, xy, xy]
      best_unit = fc.argmax()
      one_hot[:, best_unit, :, :] = 1. # targeting layers for all space      
#      one_hot[:, best_unit, xy, xy] = 1.
    else:
      raise Exception("Invalid layer type!")    
    dst.diff[:] = one_hot  

#  elif act_mode == "all":
#    dst.diff[:] = 1

        
  # Get back the gradient at the optimization layer
  diffs = net.backward(start=end, diffs=['data'])
  g = diffs['data'][0]

  grad_norm = norm(g)
  obj_act = 0

  # reset objective after each step
  dst.diff.fill(0.)

  # If grad norm is Nan, skip updating
  if math.isnan(grad_norm):
    return 1e-12, src.data[:].copy(), obj_act
  elif grad_norm == 0:
    return 0, src.data[:].copy(), obj_act

  # Check the activations
  if end in fc_layers:
    fc = acts[end][0]
    best_unit = fc.argmax()
    obj_act = fc[unit]
    
  elif end in conv_layers:
    fc = acts[end][0, :, xy, xy]
    best_unit = fc.argmax()
    obj_act = fc[unit]

  print ("max: %4s [%.2f]\t obj: %4s [%.2f]\t norm: [%.2f]" % (best_unit, fc[best_unit], unit, obj_act, grad_norm))

  # Make an update
  src.data[:] += step_size/np.abs(g).mean() * g


  #unrotation
  if not rotation == 0:
    tmp_img = src.data[0].transpose(1, 2, 0)
    tmp_img= scipy.ndimage.rotate(tmp_img, -orot, reshape=False, mode='mirror')
    src.data[0] = tmp_img.transpose(2, 0, 1)  

  #unshift
  src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)
  
  return (grad_norm, src.data[:].copy(), obj_act)


def get_shape(data_shape):

  # Return (227, 227) from (1, 3, 227, 227) tensor
  if len(data_shape) == 4:
    return (data_shape[2], data_shape[3])
  else:
    raise Exception("Data shape invalid.")


def save_image(img, name):
  '''
  Normalize and save the image.
  '''
  img = img[:,::-1, :, :] # Convert from BGR to RGB
  normalized_img = patchShow.patchShow_single(img, in_range=(-120,120))        
  scipy.misc.imsave(name, normalized_img)


def activation_maximization(net, generator, gen_in_layer, gen_out_layer, start_code, params,
                            clip=False, debug=False, unit=None, xy=0, upper_bound=None, lower_bound=None,
                            deep_dream=False, act_mode="fixed", jitter=0, rotation=0,
                            raw_ini_img=False, start_image=None, stats=False,
                            file_endstats=None, file_stats=None, file_actmax=None,
                            export=False, export_mode="exp", base_name="None"):

  # Get the input and output sizes
  data_shape = net.blobs['data'].data.shape
  generator_output_shape = generator.blobs[gen_out_layer].data.shape

  # Calculate the difference between the input image to the net being visualized
  # and the output image from the generator
  image_size = get_shape(data_shape)
  output_size = get_shape(generator_output_shape)

  # The top left offset that we start cropping the output image to get the 227x227 image
  topleft = ((output_size[0] - image_size[0])/2, (output_size[1] - image_size[1])/2)

  print ("Starting optimizing")

  x = None
  src = generator.blobs[gen_in_layer]
  
  # Make sure the layer size and initial vector size match
  assert_array_equal(src.data.shape, start_code.shape)

  # Take the starting code as the input to the generator
  src.data[:] = start_code.copy()[:]

  # Initialize an empty result
  best_xx = np.zeros(image_size)[np.newaxis]
  best_act = -sys.maxsize

  # Save the activation of each image generated
  list_acts = []

  # 0. pass the code to generator to get an image x0 for deep dream
  # todo: replace this with raw input image
  if deep_dream:
    if not raw_ini_img:
      generated = generator.forward(feat=src.data[:])
      x0 = generated[gen_out_layer]   # 256x256
    else:
      if start_image is not None:
        x0 = np.zeros((1,3,256,256), dtype='float32')        
        x0[:,:,int(topleft[0]):int(topleft[0]+image_size[0]), int(topleft[1]):int(topleft[1]+image_size[1])] = start_image.copy()
      else:
        x0 = np.zeros((1,3,256,256), dtype='float32')
        

  for o in params:
    
    # select layer
    layer = o['layer']

    for i in range(o['iter_n']):

      step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']
      
      # 1. pass the code to generator to get an image x0
      if not deep_dream:
        generated = generator.forward(feat=src.data[:])
        x0 = generated[gen_out_layer]   # 256x256

      # Crop from 256x256 to 227x227
      cropped_x0 = x0.copy()[:,:,int(topleft[0]):int(topleft[0]+image_size[0]), int(topleft[1]):int(topleft[1]+image_size[1])]

      # 2. forward pass the image x0 to net to maximize an unit k
      # 3. backprop the gradient from net to the image to get an updated image x
      stats_step = []
      actmax_step = []      
        
      grad_norm_net, x, act = make_step_net(net=net, end=layer, unit=unit, image=cropped_x0, xy=xy, step_size=step_size, act_mode=act_mode, jitter=jitter, rotation=rotation,
                                            stats=stats, stats_step=stats_step, actmax_step=actmax_step)

      # Save the solution
      # Note that we're not saving the solutions with the highest activations
      # Because there is no correlation between activation and recognizability
      best_xx = cropped_x0.copy()
      best_act = act

      # 4. Place the changes in x (227x227) back to x0 (256x256)
      updated_x0 = x0.copy()        
      updated_x0[:,:,int(topleft[0]):int(topleft[0]+image_size[0]), int(topleft[1]):int(topleft[1]+image_size[1])] = x.copy()

      # 5. backprop the image to generator to get an updated code
      if not deep_dream:      
        grad_norm_generator, updated_code = make_step_generator(net=generator, x=updated_x0, x0=x0, 
          start=gen_in_layer, end=gen_out_layer, step_size=step_size)
      else:
        grad_norm_generator = 0.5
        updated_code = 0.5
        x0 = updated_x0
        
      # Clipping code
      if clip:
        updated_code = np.clip(updated_code, a_min=-1, a_max=1) # VAE prior is within N(0,1)

      # Clipping each neuron independently
      elif upper_bound is not None:
        updated_code = np.maximum(updated_code, lower_bound) 
        updated_code = np.minimum(updated_code, upper_bound) 

      # L2 on code to make the feature vector smaller every iteration
      if o['L2'] > 0 and o['L2'] < 1:
        updated_code[:] *= o['L2']

      # Update code
      src.data[:] = updated_code

      
      # Print x every 10 iterations
      if debug:
        print (" > %s " % i)
        name = "./debug/%s.jpg" % str(i).zfill(3)

        save_image(x.copy(), name)

        # Save acts for later
        list_acts.append( (name, act) )

      if export:
        if export_mode == 'exp':
          # image generation experiment
          if i == 10 or i == 50 or i == 100 or i == 1000:
            name = "./export/" + base_name + "_" + str(i).zfill(3) + ".jpg"          
            save_image(x.copy(), name)
          
        elif export_mode == 'validation':
          # validation          
          if i == 10 or i == 100 or i == 1000:
            name = "./export/" + base_name + "_" + str(i).zfill(3) + ".jpg"          
            save_image(x.copy(), name)

        elif export_mode == 'interview':
          #for interview        
          if i == 5 or i == 10 or i == 50 or i == 100 or i == 200  or i == 400  or i == 600  or i == 800  or i == 1000:
            name = "./export/" + base_name + "_" + str(unit) + "_" + str(i).zfill(3) + ".jpg"
            save_image(x.copy(), name)            

      if stats:
        for item in actmax_step:
          file_actmax.write("%s, " % item)
        file_actmax.write("\n")

        if i == 0:
          for n in range(len(stats_step)):
            file_stats.write("%s, " % (n+1) )            
          
        for item in stats_step:
          file_stats.write("%s, " % item)
        file_stats.write("\n")        
          
      # Stop if grad is 0
      if grad_norm_generator == 0:
        print (" grad_norm_generator is 0")
        break
      elif grad_norm_net == 0:
        print (" grad_norm_net is 0")
        break

  # returning the resulting image
  print (" -------------------------")
  print (" Result: obj act [%s] " % best_act)

  if debug:
    print ("Saving list of activations...")
    for p in list_acts:
      name = p[0]
      act = p[1]

      # don't need labels
      #write_label(name, act)
      
  if stats:
    for item in actmax_step:
      file_endstats.write("%s, " % item)
    file_endstats.write("\n")
    
  return best_xx


def write_label(filename, act):
  # Add activation below each image via ImageMagick
  subprocess.call(["convert %s -gravity south -splice 0x10 %s" % (filename, filename)], shell=True)
  subprocess.call(["convert %s -append -gravity Center -pointsize %s label:\"%.2f\" -bordercolor white -border 0x0 -append %s" %
         (filename, 30, act, filename)], shell=True)


def main():

  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--unit', metavar='unit', type=int, help='an unit to visualize e.g. [0, 999]')
  parser.add_argument('--n_iters', metavar='iter', type=int, default=10, help='Number of iterations')
  parser.add_argument('--L2', metavar='w', type=float, default=1.0, nargs='?', help='L2 weight')
  parser.add_argument('--start_lr', metavar='lr', type=float, default=2.0, nargs='?', help='Learning rate')
  parser.add_argument('--end_lr', metavar='lr', type=float, default=-1.0, nargs='?', help='Ending Learning rate')
  parser.add_argument('--seed', metavar='n', type=int, default=0, nargs='?', help='Learning rate')
  parser.add_argument('--xy', metavar='n', type=int, default=0, nargs='?', help='Spatial position for conv units')
  parser.add_argument('--opt_layer', metavar='s', type=str, help='Layer at which we optimize a code')
  parser.add_argument('--act_layer', metavar='s', type=str, default="fc8", help='Layer at which we activate a neuron')
  parser.add_argument('--init_file', metavar='s', type=str, default="None", help='Init image')
  parser.add_argument('--debug', metavar='b', type=int, default=0, help='Print out the images or not')
  parser.add_argument('--clip', metavar='b', type=int, default=0, help='Clip out within a code range')
  parser.add_argument('--bound', metavar='b', type=str, default="", help='The file to an array that is the upper bound for activation range')
  parser.add_argument('--output_dir', metavar='b', type=str, default=".", help='Output directory for saving results')
  parser.add_argument('--net_weights', metavar='b', type=str, default=settings.net_weights, help='Weights of the net being visualized')
  parser.add_argument('--net_definition', metavar='b', type=str, default=settings.net_definition, help='Definition of the net being visualized')
  parser.add_argument('--gen_weights', metavar='b', type=str, default=settings.generator_weights, help='Weights of the generator network')
  parser.add_argument('--gen_definition', metavar='b', type=str, default=settings.generator_definition, help='Definition of the generator network')  
  parser.add_argument('--deep_dream', metavar='d', type=int, default=0, help='only using perceptual nets') 
  parser.add_argument('--act_mode', metavar='a', type=str, default="fixed", help='activation maximisation mode')
  parser.add_argument('--jitter', metavar='j', type=int, default=0, help='jitter in perceptual forward path')
  parser.add_argument('--rotation', metavar='r', type=int, default=0, help='rotation in perceptual forward path')
  parser.add_argument('--raw_ini_img', metavar='r', type=int, default=0, help='use raw image as initial input for deep dream')
  parser.add_argument('--stats', metavar='s', type=int, default=0, help='export stats')
  parser.add_argument('--export', metavar='e', type=int, default=0, help='export images')
  parser.add_argument('--export_mode', metavar='e', type=str, default=settings.net_definition, help='export mode [exp | validation | interview]')            
  args = parser.parse_args()

  # Default to constant learning rate
  if args.end_lr < 0:
    args.end_lr = args.start_lr

  # which neuron to visualize
  print ("-------------")
  print (" unit: %s  xy: %s" % (args.unit, args.xy))
  print (" n_iters: %s" % args.n_iters)
  print (" L2: %s" % args.L2)
  print (" start learning rate: %s" % args.start_lr)
  print (" end learning rate: %s" % args.end_lr)
  print (" seed: %s" % args.seed)
  print (" opt_layer: %s" % args.opt_layer)
  print (" act_layer: %s" % args.act_layer)
  print (" init_file: %s" % args.init_file)
  print (" clip: %s" % args.clip)
  print (" bound: %s" % args.bound)
  print ("-------------")
  print (" debug: %s" % args.debug)
  print (" output dir: %s" % args.output_dir)
  print (" net weights: %s" % args.net_weights)
  print (" net definition: %s" % args.net_definition)
  print (" gen weights: %s" % args.gen_weights)
  print (" gen definition: %s" % args.gen_definition)  
  print (" deep dream: %s" % args.deep_dream)
  print (" activation mode: %s" % args.act_mode)
  print (" jitter: %s" % args.jitter)
  print (" rotation: %s" % args.rotation)
  print (" raw initial image: %s" % args.raw_ini_img)
  print (" stats: %s" % args.stats)
  print (" export: %s" % args.export)
  print (" export mode: %s" % args.export_mode)      
  print ("-------------")

  params = [
    {
      'layer': args.act_layer,
      'iter_n': args.n_iters,
      'L2': args.L2,
      'start_step_size': args.start_lr,
      'end_step_size': args.end_lr
    }
  ]

  
  # networks
  generator = caffe.Net(args.gen_definition, args.gen_weights, caffe.TEST)
  net = caffe.Classifier(args.net_definition, args.net_weights,
               mean = mean, # ImageNet mean
               channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

  
  # input / output layers in generator
  gen_in_layer = "feat"
  gen_out_layer = "deconv0"

  # shape of the code being optimized
  shape = generator.blobs[gen_in_layer].data.shape

  # Fix the seed
  np.random.seed(args.seed)

  start_image = None
  base_name = "None"
  if args.init_file != "None":
    start_code, start_image = get_code(args.init_file, args.opt_layer)
    print(start_image.shape)
    print ("Loaded start code: ", start_code.shape)

    base_name = os.path.splitext(os.path.basename(args.init_file))[0]    
  else:
    start_code = np.random.normal(0, 1, shape)

  # Load the activation range
  upper_bound = lower_bound = None

  # Set up clipping bounds
  if args.bound != "":
    n_units = shape[1]
    upper_bound = np.loadtxt(args.bound, delimiter=' ', usecols=np.arange(0, n_units), unpack=True)
    upper_bound = upper_bound.reshape(start_code.shape)

    # Lower bound of 0 due to ReLU
    lower_bound = np.zeros(start_code.shape)

  if args.deep_dream:
    gen_mode = "DCNN"
  else:
    gen_mode = "DGN"
    
  # Save stat file
  stat_filename = "%s_%s_%s_%s_%s_%s_%s_%s__%s.txt" % (
    gen_mode,
    args.act_mode,
    args.act_layer, 
    str(args.unit).zfill(4),
    os.path.basename(args.init_file),    
    str(args.n_iters).zfill(2), 
    args.L2, 
    args.start_lr,
    args.seed
  )
    
  if args.stats:
    file_endstats = open("stats/endstats.txt", 'a')
    file_stats = open("stats/stats_" + stat_filename, 'w')
    file_actmax = open("stats/actmax_" + stat_filename, 'w')        
  else:
    file_endstats = None
    file_stats = None
    file_actmax = None
    
  # Optimize a code via gradient ascent
  output_image = activation_maximization(net, generator, gen_in_layer, gen_out_layer, start_code, params, 
                                         clip=args.clip, unit=args.unit, xy=args.xy, debug=args.debug,
                                         upper_bound=upper_bound, lower_bound=lower_bound,
                                         deep_dream=args.deep_dream, act_mode=args.act_mode, jitter=args.jitter, rotation=args.rotation,
                                         raw_ini_img=args.raw_ini_img, start_image=start_image,
                                         stats=args.stats, file_endstats=file_endstats, file_stats=file_stats, file_actmax=file_actmax,
                                         export=args.export, export_mode=args.export_mode, base_name=base_name)

         
    
  # Save image
  filename = "%s/%s_%s_%s_%s_%s_%s_%s_%s__%s.jpg" % (
    args.output_dir,
    gen_mode,
    args.act_mode,
    args.act_layer, 
    str(args.unit).zfill(4),
    os.path.basename(args.init_file),    
    str(args.n_iters).zfill(2), 
    args.L2, 
    args.start_lr,
    args.seed
  )


  # Save image
  save_image(output_image, filename)
  print ("Saved to %s" % filename)

  if args.debug:
    save_image(output_image, "./debug/%s.jpg" % str(args.n_iters).zfill(3))

if __name__ == '__main__':
  main()
