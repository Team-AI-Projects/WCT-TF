from __future__ import division, print_function


import os
import math
import argparse
import numpy as np
import tensorflow as tf
from utils import preserve_colors_np
from utils import get_files, get_img, get_img_crop, save_img, resize_to, center_crop, remaster_pic
import scipy
# from glob import glob
import time
from wct import WCT


parser = argparse.ArgumentParser()

parser.add_argument('--checkpoints', nargs='+', type=str, help='List of checkpoint directories', required=True)
parser.add_argument('--relu-targets', nargs='+', type=str, help='List of reluX_1 layers, corresponding to --checkpoints', required=True)
parser.add_argument('--vgg-path', type=str, help='Path to vgg_normalised.t7', default='models/vgg_normalised.t7')

parser.add_argument('--content-path', type=str, dest='content_path', help='Content image or folder of images')
parser.add_argument('--style-path', type=str, dest='style_path', help='Style image or folder of images')

parser.add_argument('--out-path', type=str, dest='out_path', help='Output folder path')

parser.add_argument('--keep-colors', action='store_true', help="Preserve the colors of the style image", default=False)
parser.add_argument('--device', type=str, help='Device to perform compute on, e.g. /gpu:0', default='/gpu:0')

parser.add_argument('--style-size', type=int, help="Resize style image to this size before cropping, default 512", default=0)
parser.add_argument('--crop-size', type=int, help="Crop square size, default 256", default=0)
parser.add_argument('--content-size', type=int, help="Resize short side of content image to this", default=0)
parser.add_argument('--passes', type=int, help="# of stylization passes per content image", default=1)

parser.add_argument('-r','--random', type=int, help="Choose # of random subset of images from style folder", default=0)

parser.add_argument('--alpha', type=float, help="Alpha blend value", default=1)
parser.add_argument('--concat', action='store_true', help="Concatenate style image and stylized output", default=False)
parser.add_argument('--adain', action='store_true', help="Use AdaIN instead of WCT", default=False)

# parser.add_argument('-p','--portrait-mode', nargs='+',type=str,help="0 = Do nothing. 1 = Only stylize the subject/foreground. 2 = Only stylize background. 3 = Stylize subject and replace background with color (input hexcode e.g. #00CCFF).",default=0)

parser.add_argument('-rm','--remaster', action='store_true',help="Remaster and resize pic. Normally makes it look like watercolor.",default=False)

parser.add_argument('--mask', type=str,help="Use a mask file or dir of masks. Works best with only one content image and one mask folder",default=False)
## Style swap args
parser.add_argument('--swap5', action='store_true', help="Swap style on layer relu5_1", default=False)
parser.add_argument('--ss-alpha', type=float, help="Style swap alpha blend", default=0.6)
parser.add_argument('--ss-patch-size', type=int, help="Style swap patch size", default=3)
parser.add_argument('--ss-stride', type=int, help="Style swap stride", default=1)

args = parser.parse_args()


def main():
    # start = time.time()
    # Load the WCT model
    wct_model = WCT(checkpoints=args.checkpoints, 
                                relu_targets=args.relu_targets,
                                vgg_path=args.vgg_path, 
                                device=args.device,
                                ss_patch_size=args.ss_patch_size, 
                                ss_stride=args.ss_stride)

    # Get content & style full paths
    if os.path.isdir(args.content_path):
        content_files = get_files(args.content_path)
    else: # Single image file
        content_files = [args.content_path]
    if os.path.isdir(args.style_path):
        style_files = get_files(args.style_path)
        if args.random > 0:
            style_files = np.random.choice(style_files, args.random)
    else: # Single image file
        style_files = [args.style_path]

    os.makedirs(args.out_path, exist_ok=True)

    thetotal = len(content_files) * len(style_files)

    count = 1

    ### Apply each style to each content image
    for content_fullpath in content_files:
        content_prefix, content_ext = os.path.splitext(content_fullpath)
        content_prefix = os.path.basename(content_prefix)  # Extract filename prefix without ext

        content_img = get_img(content_fullpath)
        if args.content_size > 0:
            content_img = resize_to(content_img, args.content_size)
        
        for style_fullpath in style_files: 

            style_prefix, _ = os.path.splitext(style_fullpath)
            style_prefix = os.path.basename(style_prefix)  # Extract filename prefix without ext

            if args.mask:
                mask_prefix_, _ = os.path.splitext(args.mask)
                mask_prefix = os.path.basename(mask_prefix_)

            if args.keep_colors:
                style_prefix = "KPT_"+style_prefix
            if args.concat:
                style_prefix = "CON_"+style_prefix
            if args.adain:
                style_prefix = "ADA_"+style_prefix
            if args.swap5:
                style_prefix = "SWP_"+style_prefix
            if args.mask:
                style_prefix = "MSK_"+mask_prefix+'_'+style_prefix
            if args.remaster:
                style_prefix = style_prefix + '_REMASTERED'

            out_f = os.path.join(args.out_path, '{}_{}{}'.format(content_prefix, style_prefix, content_ext))

            if os.path.isfile(out_f):
                print("SKIP" , out_f)
                count += 1
                continue

            # style_img = get_img_crop(style_fullpath, resize=args.style_size, crop=args.crop_size)
            # style_img = resize_to(get_img(style_fullpath), content_img.shape[0])
            style_img = get_img(style_fullpath)
            if style_img == ("IMAGE IS BROKEN"):
                continue

            if args.style_size > 0:
                style_img = resize_to(style_img, args.style_size)
            if args.crop_size > 0:
                style_img = center_crop(style_img, args.crop_size)
            if args.keep_colors:
                style_img = preserve_colors_np(style_img, content_img)
            
            # if args.noise:  # Generate textures from noise instead of images
            #     frame_resize = np.random.randint(0, 256, frame_resize.shape, np.uint8)
            #     frame_resize = gaussian_filter(frame_resize, sigma=0.5)

            # Run the frame through the style network
            stylized_rgb = wct_model.predict(content_img, style_img, args.alpha, args.swap5, args.ss_alpha, args.adain)

            if args.passes > 1:
                for _ in range(args.passes-1):
                    stylized_rgb = wct_model.predict(stylized_rgb, style_img, args.alpha, args.swap5, args.ss_alpha, args.adain)
            
            if args.mask:
                import cv2
                cv2.imwrite('./tmp.png',stylized_rgb)
                stylized_rgb = cv2.imread('./tmp.png')

                mask = cv2.imread(args.mask,cv2.IMREAD_GRAYSCALE)

                # from scipy.misc import bytescale
                # mask = bytescale(mask)
                # mask = scipy.ndimage.imread(args.mask,flatten=True,mode='L')
                height, width = stylized_rgb.shape[:2]
                # print(height, width)

                # Resize the mask to fit the image.
                mask = scipy.misc.imresize(mask, (height,width), interp='bilinear')
                stylized_rgb = cv2.bitwise_and(stylized_rgb,stylized_rgb,mask=mask)
            

            # Stitch the style + stylized output together, but only if there's one style image
            if args.concat:
                # Resize style img to same height as frame
                # style_prefix = style_prefix + "CON_"
                # content_img_resized = scipy.misc.imresize(content_img, (stylized_rgb.shape[0], stylized_rgb.shape[0]))
                style_img_resized   = scipy.misc.imresize(style_img,   (stylized_rgb.shape[0], stylized_rgb.shape[0]))
                # margin = np.ones((style_img_resized.shape[0], 10, 3)) * 255
                stylized_rgb = np.hstack([style_img_resized, stylized_rgb])

            # Format for out filename: {out_path}/{content_prefix}_{style_prefix}.{content_ext}
            # out_f = f'{content_prefix}_{style_prefix}.{content_ext}'
            out_f = os.path.join(args.out_path, '{}_{}{}'.format(content_prefix, style_prefix, content_ext))

            if args.remaster:
                # outf = os.path.join(args.out_path, '{}_{}_REMASTERED{}'.format(content_prefix, style_prefix, content_ext))
                stylized_rgb = remaster_pic(stylized_rgb)
            save_img(out_f, stylized_rgb)
            totalfiles = len([name for name in os.listdir(args.out_path) if os.path.isfile(os.path.join(args.out_path, name))])
            # percent = math.floor(float(totalfiles/thetotal))
            print("{}/{} TOTAL FILES".format(count,thetotal))
            count += 1
            print("{}: Wrote stylized output image to {}".format(count, out_f))

    # print("Finished stylizing {} outputs in {}s  - {}m  - {}h".format(count, time.time() - start, (time.time()-start)/60), (time.time()-start)/3600)

if __name__ == '__main__':
    main()
