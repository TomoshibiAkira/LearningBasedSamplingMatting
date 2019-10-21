import os
import logging
import time

import torch
import cv2
import numpy as np
import torch.utils.data
import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from imgaug import parameters as iap
import torchvision.transforms as transforms

class DIMTrainSet(torch.utils.data.Dataset):
    def __init__(self, data_root, isTrain=True, image_shape=[480, 848]):
        self.data_root = data_root
        self.image_shape = image_shape

        if isTrain:
            set_file_path = os.path.join(data_root, 'train_set.txt')
        else:
            set_file_path = os.path.join(data_root, 'val_set.txt')
        set_file = open(set_file_path, 'r')
        self.data_list = []

        for line in set_file:
            line = line.strip()
            rgb_path, alpha_path, trimap_path, fg_path, bg_path = line.split(' ')
            self.data_list.append({
                'rgb':rgb_path,
                'alpha': alpha_path,
                'trimap': trimap_path,
                'fg': fg_path,
                'bg': bg_path
            })


    def __len__(self):
        return len(self.data_list)

    def load_image(self, img_path, shape=None):
        img = cv2.imread(img_path)
        if shape is not None:
            img = cv2.resize(img, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32)
        img = img / 127.5 - 1.0
        img = np.transpose(img, (2,0,1)) # C,H,W

        return torch.from_numpy(img).float()

    def load_trimap(self, trimap_path, shape=None):
        trimap = cv2.imread(trimap_path, 0)
        if shape is not None:
            trimap = cv2.resize(trimap, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        small_trimap = cv2.resize(trimap, dsize=(shape[1]//8, shape[0]//8), interpolation=cv2.INTER_NEAREST)

        trimask = np.uint8(np.logical_and(trimap<250, trimap>10))
        trimask = trimask[np.newaxis, ...]

        trimap = np.uint8(trimap > 0)
        small_trimap = np.uint8(small_trimap > 0)

        trimap = trimap[np.newaxis, ...] # 1, H, W
        small_trimap = small_trimap[np.newaxis, ...]

        return torch.from_numpy(trimap).float(), torch.from_numpy(small_trimap).float(), torch.from_numpy(trimask).float()



    def __getitem__(self, idx):
        # norm: img/127 - 1.0
        rgb_path = os.path.join(self.data_root, self.data_list[idx]['rgb'])
        bg_path = os.path.join(self.data_root, self.data_list[idx]['bg'])
        trimap_path = os.path.join(self.data_root, self.data_list[idx]['trimap'])

        rgb = self.load_image(rgb_path, shape=self.image_shape)
        bg = self.load_image(bg_path, shape=self.image_shape)

        trimap, small_trimap, trimask = self.load_trimap(trimap_path, shape=self.image_shape)

        return rgb, trimap, small_trimap, trimask, bg

class DIMHeavyComposeSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_shape=(256, 256), isTrain=True, silence=False):
        self.silence = silence
        self.data_root = data_root
        self.is_train = isTrain
        self.image_shape = (image_shape[0], image_shape[1])

        if isTrain:
            fg_set_file_path = os.path.join(data_root, 'fg_train_set.txt')
            bg_set_file_path = os.path.join(data_root, 'bg_train_set.txt')
        else:
            fg_set_file_path = os.path.join(data_root, 'fg_val_set.txt')
            bg_set_file_path = os.path.join(data_root, 'bg_val_set.txt')

        fg_file = open(fg_set_file_path, 'r')
        bg_file = open(bg_set_file_path, 'r')

        self.fg_list = []
        self.bg_list = []
        self.alpha_list = []

        for line in fg_file:
            line = line.strip()
            fg_path, alpha_path = line.split(' ')
            self.fg_list.append(fg_path)
            self.alpha_list.append(alpha_path)

        for line in bg_file:
            bg_path = line.strip()
            self.bg_list.append(bg_path)

        self.num_fg = len(self.fg_list)
        self.num_bg = len(self.bg_list)
        if not isTrain:
            assert self.num_fg == self.num_bg, "len(fg_list) must equal to len(bg_list) in eval mode. {}!={}".format(len(self.fg_list), len(self.bg_list))

        sometimes = lambda aug: iaa.Sometimes(0.3, aug)

        # apply to fg, bg
        self.color_aug = iaa.Sequential([
            iaa.MultiplyHueAndSaturation(mul=iap.TruncatedNormal(1.0, 0.2, 0.5, 1.5)), # mean, std, low, high
            iaa.GammaContrast(gamma=iap.TruncatedNormal(1.0, 0.2, 0.5, 1.5)),
            iaa.AddToHue(value=iap.TruncatedNormal(0.0, 0.1*100, -0.2*255, 0.2*255))
        ])

        # apply to compose
        self.compose_shape_aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            # iaa.OneOf([
            #     iaa.Resize({"height":480, "width": "keep-aspect-ratio"}),
            #     iaa.Resize({"height":640, "width": "keep-aspect-ratio"}),
            #     # iaa.Resize({"height": 800, "width": "keep-aspect-ratio"})
            # ]),
            # iaa.Resize(size=(512, 512)),
            iaa.OneOf([
                # iaa.CropToFixedSize(width=256, height=256),
                iaa.CropToFixedSize(width=384, height=384),
                iaa.CropToFixedSize(width=480, height=480),
                iaa.CropToFixedSize(width=512, height=512),
                iaa.CropToFixedSize(width=640, height=640),
                iaa.CropToFixedSize(width=800, height=800)
            ]),
            sometimes(iaa.PadToFixedSize(width=512, height=512, pad_mode="constant", pad_cval=0)), # TODO:
            iaa.Resize({"height": self.image_shape[0], "width":self.image_shape[1]})
        ])
        self.fg_simple_aug = iaa.Sequential([iaa.Fliplr(0.5)])

        self.bg_aug = iaa.Sequential([
            iaa.CropToFixedSize(width=self.image_shape[1], height=self.image_shape[0]),

            # iaa.OneOf([
            #     # iaa.CropToFixedSize(width=256, height=256),
            #     iaa.CropToFixedSize(width=384, height=384),
            #     iaa.CropToFixedSize(width=self.image_shape[1], height=self.image_shape[0]),
            #     iaa.CropToFixedSize(width=480, height=480),
            #     iaa.CropToFixedSize(width=512, height=512),
            #     # iaa.CropToFixedSize(width=640, height=640),
            # ]),

            iaa.Resize({"height": self.image_shape[0], "width": self.image_shape[1]}),
            iaa.GammaContrast(gamma=iap.TruncatedNormal(1.0, 0.3, 0.5, 1.5)),
            iaa.MultiplySaturation(mul=iap.TruncatedNormal(1.0, 0.3, 0.5, 1.5)),
            # iaa.GaussianBlur(sigma=(0.0, 1.5)),
            iaa.Fliplr(0.5),
        ])

        self.scale_down = iaa.Sequential([
            iaa.Resize(0.8),
            iaa.PadToFixedSize(width=self.image_shape[1], height=self.image_shape[0], pad_mode='constant', pad_cval=0)
        ])

    def resize_keep_ratio(self, img, height, width, alpha=None):
        ori_height = img.shape[0]
        ori_width = img.shape[1]

        ratio1 = height*1.0 / width
        ratio2 = ori_height*1.0 / ori_width

        if ratio1 > ratio2:
            # new_width = width
            new_width = width
            new_height = int((new_width*1.0/width)*ori_height)
        else:
            new_height = height
            new_width = int((new_height*1.0/height) * ori_width)
        img = cv2.resize(img, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)
        if alpha is None:
            return img
        alpha = cv2.resize(alpha, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)

        return img, alpha

    def compose_fg_bg(self, fg, alpha, bg):
        alpha3 = alpha[..., np.newaxis]
        compose = alpha3 * fg + (1.0 - alpha3) * bg
        return compose

    def load_alpha(self, alpha_path):
        alpha = cv2.imread(alpha_path, 0)
        alpha = np.where(alpha<10, 0, alpha)
        alpha = np.where(alpha>245, 255, alpha)
        return np.clip(np.float32(alpha/255.0), 0, 1.0)

    def vis_2fg(self, fg1, alpha1, fg2, alpha2, compose, bg):
        # not real fg color, just for visualization
        alpha2 = alpha2[..., np.newaxis]
        alpha1 = alpha1[..., np.newaxis]
        divide_alpha = (alpha1+alpha2)-alpha1*alpha2
        divide_alpha = np.where(divide_alpha<0.01, 1.0, divide_alpha)
        fake_fg = fg2 * alpha2 + (1.0 - alpha2) * fg1
        new_fg = np.where(divide_alpha>0.01, (compose - (1.0-alpha1)*(1.0-alpha2)*bg) / divide_alpha, fake_fg)
        return new_fg

    def random_crop_fg_on_transition(self, image, alpha, crop_size_choices=[384, 480, 512, 640, 800]):
        # TODO: aug
        temp_alpha = alpha.copy()

        # avoid crop center on image border
        border_threshold = 100
        temp_alpha[0:border_threshold, :] = 256
        temp_alpha[:, 0:border_threshold] = 256
        temp_alpha[-border_threshold:-1, :] = 256
        temp_alpha[:, -border_threshold:-1] = 256

        trans_h_list, trans_w_list = np.where(np.logical_and(temp_alpha>0.01, temp_alpha<0.99))

        while True:
            rand_center_id = np.random.randint(0, len(trans_h_list))
            center_h = trans_h_list[rand_center_id]
            center_w = trans_w_list[rand_center_id]

            crop_size = np.random.choice(crop_size_choices)

            h_begin = max(center_h-crop_size//2, 0)
            h_end = min(h_begin+crop_size, image.shape[0])
            w_begin = max(center_w-crop_size//2, 0)
            w_end = min(w_begin+crop_size, image.shape[1])

            new_image = image[h_begin:h_end, w_begin:w_end, :]
            new_alpha = alpha[h_begin:h_end, w_begin:w_end]

            # assert transition_retion_area>0
            if np.sum(np.logical_and(new_alpha>0.01, new_alpha<0.99)) > 0:
                break

        new_image = cv2.resize(new_image, dsize=(self.image_shape[1], self.image_shape[0]), interpolation=cv2.INTER_CUBIC)
        new_alpha = cv2.resize(new_alpha, dsize=(self.image_shape[1], self.image_shape[0]), interpolation=cv2.INTER_CUBIC)

        return new_image, np.clip(new_alpha, 0, 1.0)





    def get_train_sample(self, fg_ind, bg_ind, second_fg_ind=-1):
        first_fg_path = os.path.join(self.data_root, self.fg_list[fg_ind])
        first_alpha_path = os.path.join(self.data_root, self.alpha_list[fg_ind])
        bg_path = os.path.join(self.data_root, self.bg_list[bg_ind])

        def check_path(pp):
            if not os.path.isfile(pp):
                print("not exist ", pp)
                exit(0)
        # check_path(first_fg_path)
        # check_path(first_alpha_path)
        # check_path(bg_path)

        fg = cv2.imread(first_fg_path)
        alpha = self.load_alpha(first_alpha_path)
        bg = cv2.imread(bg_path)

        fg = self.color_aug(image=fg)

        heat_map_alpha = HeatmapsOnImage(alpha, shape=fg.shape, min_value=0.0, max_value=1.0)
        # fg, aug_alpha = self.compose_shape_aug(image=fg, heatmaps=heat_map_alpha)
        fg, aug_alpha = self.fg_simple_aug(image=fg, heatmaps=heat_map_alpha)
        alpha = np.squeeze(aug_alpha.get_arr())
        fg, alpha = self.random_crop_fg_on_transition(fg, alpha)

        # to avoid all the pixels are masked, scale down the large fg objects
        while np.sum(alpha<0.01) < 100:
            # print("warning: fg-{} mask is to large. mask_area={}".format(fg_ind, np.sum(alpha<0.01)))
            heat_map_alpha = HeatmapsOnImage(alpha, shape=fg.shape, min_value=0.0, max_value=1.0)
            fg, aug_alpha = self.scale_down(image=fg, heatmaps=heat_map_alpha)
            alpha = np.squeeze(aug_alpha.get_arr())


        # compose fg bg
        bg = self.bg_aug(image=bg)
        # bg = cv2.resize(bg, dsize=(fg.shape[1], fg.shape[0]), interpolation=cv2.INTER_CUBIC)
        # TODO: augment bg
        compose = self.compose_fg_bg(fg, alpha, bg)

        if second_fg_ind >= 0:
            second_fg_path = os.path.join(self.data_root, self.fg_list[second_fg_ind])
            second_alpha_path = os.path.join(self.data_root, self.alpha_list[second_fg_ind])

            fg2 = cv2.imread(second_fg_path)
            fg2 = self.color_aug(image=fg2)

            alpha2 = self.load_alpha(second_alpha_path)
            # second_fg, second_alpha = self.resize_keep_ratio(second_fg, fg.shape[0], fg.shape[1], alpha=second_alpha)
            heat_map_alpha = HeatmapsOnImage(alpha2, shape=fg2.shape, min_value=0.0, max_value=1.0)
            # fg2, aug_alpha2 = self.compose_shape_aug(image=fg2, heatmaps=heat_map_alpha)
            fg2, aug_alpha2 = self.fg_simple_aug(image=fg2, heatmaps=heat_map_alpha)
            alpha2 = np.squeeze(aug_alpha2.get_arr()) # dsize
            fg2, alpha2 = self.random_crop_fg_on_transition(fg2, alpha2)

            while np.sum(alpha2<0.01) < 100:
                # print("warning: fg-{} mask is to large. mask_area={}".format(second_fg_ind, np.sum(alpha2 < 0.01)))
                heat_map_alpha = HeatmapsOnImage(alpha2, shape=fg2.shape, min_value=0.0, max_value=1.0)
                fg2, aug_alpha2 = self.scale_down(image=fg2, heatmaps=heat_map_alpha)
                alpha2 = np.squeeze(aug_alpha2.get_arr())


            new_alpha = (alpha + alpha2) - alpha * alpha2
            if np.sum(new_alpha<0.01)<100:
                if not self.silence:
                    print("warning: skip compose second fg. mask_area={}".format(np.sum(new_alpha<0.01)))
            else:

                compose = self.compose_fg_bg(fg2, alpha2, compose)
                fg = self.vis_2fg(fg, alpha, fg2, alpha2, compose, bg)  # just for visualization # TODO: not real fg color

                alpha = (alpha + alpha2) - alpha * alpha2



        trimask = np.logical_and(alpha>0.01, alpha<0.99).astype('uint8')
        full_mask = np.uint8(alpha>0.01) # translucent+fg


        d = np.random.randint(5, 36)
        # d = 6
        # print("dilate", d)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        dilate_trimask = cv2.dilate(trimask, element)
        
        # trimask translation
        _x = np.random.randint(-d // 2+1, d // 2)
        _y = np.random.randint(-d // 2+1, d // 2)
        translation_matrix = np.float32([[1, 0, _x], [0, 1, _y]])
        dilate_trimask = cv2.warpAffine(src=dilate_trimask, M=translation_matrix,
                dsize=(dilate_trimask.shape[1], dilate_trimask.shape[0]),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

        temp_full_mask = np.where(dilate_trimask > 0, 1, full_mask) # unknown + fg
        if np.sum(temp_full_mask==0)<100:
            # print("warning: skip random dilate. area={}".format(np.sum(temp_full_mask==0)))
            temp_full_mask = np.where(trimask>0, 1, full_mask)
            dilate_trimask = trimask
        full_mask = temp_full_mask
        trimask = dilate_trimask

        trimask = trimask[..., np.newaxis]
        alpha = alpha[..., np.newaxis]


        trimap = np.where(alpha>0.99, 2, 0)
        trimap = np.where(trimask>0, 1, trimap) # (h,w,1)
        small_trimap = cv2.resize(trimap, dsize=(self.image_shape[1]//8, self.image_shape[0]//8), interpolation=cv2.INTER_NEAREST)
        small_trimap = small_trimap[..., np.newaxis]
        # print("small trimap shape", small_trimap.shape)

        compose = compose / 127.5 - 1.0
        bg = bg / 127.5 - 1.0
        fg = fg / 127.5 - 1.0

        compose = np.transpose(compose, (2,0,1))
        fg = np.transpose(fg, (2,0,1))
        bg = np.transpose(bg, (2, 0, 1))
        alpha = np.transpose(alpha, (2,0,1))

        trimap = np.transpose(trimap, (2,0,1))
        small_trimap = np.transpose(small_trimap, (2,0,1))

        # return compose, fg, bg, trimask, full_mask, small_mask, alpha, small_fg_mask, trimap, small_trimap
        return compose, fg, bg, alpha, trimap, small_trimap

    def get_eval_sample(self, fg_ind, bg_ind):
        fg_path = os.path.join(self.data_root, self.fg_list[fg_ind])
        alpha_path = os.path.join(self.data_root, self.alpha_list[fg_ind])
        bg_path = os.path.join(self.data_root, self.bg_list[bg_ind])

        fg = cv2.imread(fg_path)
        bg = cv2.imread(bg_path)
        alpha = cv2.imread(alpha_path, 0)

        fg = cv2.resize(fg, dsize=(self.image_shape[1], self.image_shape[0]), interpolation=cv2.INTER_CUBIC)
        alpha = cv2.resize(alpha, dsize=(self.image_shape[1], self.image_shape[0]), interpolation=cv2.INTER_CUBIC)
        alpha = np.where(alpha<10, 0, alpha)
        alpha = np.where(alpha>250, 255, alpha)
        bg = cv2.resize(bg, dsize=(self.image_shape[1], self.image_shape[0]), interpolation=cv2.INTER_CUBIC)

        alpha = alpha[..., np.newaxis]
        alpha = alpha / 255.0
        trimask = np.logical_and(alpha > 0.01, alpha < 0.99).astype('uint8') # optimal unknown
        full_mask = np.uint8(alpha > 0.01)  # translucent+fg


        # d = np.random.randint(1, 12)
        d = 21
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        dilate_trimask = cv2.dilate(trimask, element) # (h,w)
        dilate_trimask = dilate_trimask[..., np.newaxis]

        temp_full_mask = np.where(dilate_trimask > 0, 1, full_mask) # (h,w,1)
        if np.sum(full_mask==0) < 100:
            # print("warning: skip trimask dilation. evaluation data {} mask is too large. area={}".format(fg_ind, np.sum(full_mask==0)))
            temp_full_mask = np.where(trimask>0, 1, full_mask)
            dilate_trimask = trimask
        full_mask = temp_full_mask
        trimask = dilate_trimask


        trimap = np.where(alpha>0.99, 2, 0) # fg
        trimap = np.where(trimask>0, 1, trimap) # (h,w,1)
        small_trimap = cv2.resize(trimap, dsize=(self.image_shape[1]//8, self.image_shape[0]//8), interpolation=cv2.INTER_NEAREST)
        small_trimap = small_trimap[..., np.newaxis]
        # print("small trimap shape", small_trimap.shape)

        compose = alpha * fg + (1.0-alpha)*bg

        compose = compose / 127.5 - 1.0
        fg = fg / 127.5 - 1.0
        bg = bg / 127.5 - 1.0

        compose = np.transpose(compose, (2,0,1))
        fg = np.transpose(fg, (2,0,1))
        bg = np.transpose(bg, (2, 0, 1))
        alpha = np.transpose(alpha, (2,0,1))
        trimap = np.transpose(trimap, (2,0,1))
        small_trimap = np.transpose(small_trimap, (2,0,1))


        # return compose, fg, bg, trimask, full_mask, small_mask, alpha, small_fg_mask, trimap, small_trimap
        return compose, fg, bg, alpha, trimap, small_trimap


    def __len__(self):
        return len(self.bg_list)



    def __getitem__(self, idx):

        if self.is_train:
            fg_id = np.random.randint(0, self.num_fg)
            # fg_id = np.random.randint(0, 3)
            fg2_id = -1
            if np.random.ranf() > 0.5:
                fg2_id = np.random.randint(0, self.num_fg)
            bg_id = np.random.randint(0, self.num_bg)
            compose, fg, bg, alpha, trimap, small_trimap = self.get_train_sample(fg_id, bg_id, fg2_id)

        else:
            fg_id = idx
            bg_id = idx
            fg2_id = -1
            compose, fg, bg, alpha, trimap, small_trimap = self.get_eval_sample(fg_id, bg_id)

        if self.is_train and np.sum(trimap<0.01) < 100:
            print("Error: fg object is too large. fg_id={}, fg2_id={}, area={}".format(fg_id, fg2_id, np.sum(trimap<0.01)))
            cv2.imwrite("{}_{}_{}.jpg".format(time.strftime("%Y%m%d-%H%M"), fg_id, fg2_id), 255*np.squeeze(trimap<0.01))
            exit(0)

        assert np.sum(trimap==1) > 10, "too small trimap"


        compose = torch.from_numpy(compose).float()
        fg = torch.from_numpy(fg).float()
        bg = torch.from_numpy(bg).float()

        alpha = torch.from_numpy(alpha).float()

        trimap = torch.from_numpy(trimap).float()
        small_trimap = torch.from_numpy(small_trimap).float()

        # return compose, fg, bg, trimask, full_mask, small_mask, alpha
        return compose, fg, bg, alpha, trimap, small_trimap

def test():
    from torch.utils.data import DataLoader
    import torchvision.utils as vutils
    import torch.nn.functional as F

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    train_set = DIMHeavyComposeSet('../DIM_onthefly', image_shape=[640,640], isTrain=True)
    #train_loader = DataLoader(train_set, batch_size=4, shuffle=True,
    #                          drop_last=False, num_workers=1)# worker_init_fn=worker_init_fn)
    train_loader = DataLoader(train_set,
                          batch_size=16,
                          shuffle=True,
                          drop_last=True,
                          num_workers=args.n_threads,
                          pin_memory=True)

    np.random.seed(777)
    start = time.time()
    for batch_idx, (image, fg, bg, alpha, trimap, small_trimap) in enumerate(train_loader):
        image = torch.flip(image, dims=[1])
        # fg = torch.flip(fg, dims=[1])
        bg = torch.flip(bg, dims=[1])

        vis_trimap = F.one_hot(trimap.long(), 3).transpose(1,4).squeeze(-1)

        # print(small_fg_mask.shape)
        vutils.save_image(image, 'testdata/{}_image.jpg'.format(batch_idx), nrow=2, normalize=True)
        # # vutils.save_image(fg, 'testdata/{}_fg.jpg'.format(batch_idx), nrow=2, normalize=True)
        # vutils.save_image(bg, 'testdata/{}_bg.jpg'.format(batch_idx), nrow=2, normalize=True)
        # vutils.save_image(trimask, 'testdata/{}_trimask.png'.format(batch_idx), nrow=2, normalize=True)
        # vutils.save_image(mask, 'testdata/{}_alpha.png'.format(batch_idx), nrow=2, normalize=True)
        vutils.save_image(vis_trimap*255, 'testdata/{}_trimap.png'.format(batch_idx), nrow=2, normalize=True)
        # print(batch_idx, image.shape, trimask.shape)

        print("{}/{}".format(batch_idx, len(train_loader)), end='\r')
        
        if(batch_idx > 100):
            end = time.time()
            break
    print("time/batch: {}".format((end-start)/100))

if __name__ == '__main__':
    test()
