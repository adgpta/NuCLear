import cv2 as cv
import numpy as np
import tifffile
import os
import fastremap
from tqdm import tqdm
from scipy.ndimage import find_objects

class mask_overlay():
    def __init__(self):
        True
        self.image_path = None

    def imread(self, filename):
        """ read in image with tif or image file type supported by cv2 """
        # ensure that extension check is not case sensitive
        ext = os.path.splitext(filename)[-1].lower()
        if ext== '.tif' or ext=='.tiff':
            with tifffile.TiffFile(filename) as tif:
                ltif = len(tif.pages)
                try:
                    full_shape = tif.shaped_metadata[0]['shape']
                except:
                    try:
                        page = tif.series[0][0]
                        full_shape = tif.series[0].shape
                    except:
                        ltif = 0
                if ltif < 10:
                    img = tif.asarray()
                else:
                    page = tif.series[0][0]
                    shape, dtype = page.shape, page.dtype
                    ltif = int(np.prod(full_shape) / np.prod(shape))
                    print(f'GUI_INFO: reading tiff with {ltif} planes..')
                    img = np.zeros((ltif, *shape), dtype=dtype)
                    for i,page in enumerate(tqdm(tif.series[0])):
                        img[i] = page.asarray()
                    img = img.reshape(full_shape)            
            return img

    def _initialize_image(self, image):
        #this function comes right after "imread"
        #returns image(stack) and length of image(stack) or NZ
        """ format image for GUI """
        self.onechan=False
        if image.ndim > 3:
            # make tiff Z x channels x W x H
            if image.shape[0]<4:
                # tiff is channels x Z x W x H
                image = np.transpose(image, (1,0,2,3))
            elif image.shape[-1]<4:
                # tiff is Z x W x H x channels
                image = np.transpose(image, (0,3,1,2))
            # fill in with blank channels to make 3 channels
            if image.shape[1] < 3:
                shape = image.shape
                image = np.concatenate((image,
                                np.zeros((shape[0], 3-shape[1], shape[2], shape[3]), dtype=np.uint8)), axis=1)
                if 3-shape[1]>1:
                    self.onechan=True
            image = np.transpose(image, (0,2,3,1))
        elif image.ndim==3:
            if image.shape[0] < 5:
                image = np.transpose(image, (1,2,0))
            if image.shape[-1] < 3:
                shape = image.shape
                #if parent.autochannelbtn.isChecked():
                #    image = normalize99(image) * 255
                image = np.concatenate((image,np.zeros((shape[0], shape[1], 3-shape[2]),dtype=type(image[0,0,0]))), axis=-1)
                if 3-shape[2]>1:
                    self.onechan=True
                image = image[np.newaxis,...]
            elif image.shape[-1]<5 and image.shape[-1]>2:
                image = image[:,:,:3]
                #if parent.autochannelbtn.isChecked():
                #    image = normalize99(image) * 255
                image = image[np.newaxis,...]
        else:
            image = image[np.newaxis,...]
        
        img_min = image.min() 
        img_max = image.max()
        stack = image
        NZ = len(stack)
        stack = stack.astype(np.float32)
        stack -= img_min
        if img_max > img_min + 1e-3:
            stack /= (img_max - img_min)
        stack *= 255
        if NZ>1:
            print('GUI_INFO: converted to float and normalized values to 0.0->255.0')
        if stack.ndim < 4:
            self.onechan=True
            stack = stack[:,:,:,np.newaxis]

        return stack, NZ

    def _load_masks(self, masks):
        """ load zeros-based masks (0=no cell, 1=cell 1, ...) """
        self.outlines = None
        if masks.ndim>3:
            # Z x nchannels x Ly x Lx
            if masks.shape[-1]>5:
                self.outlines = masks[...,1]
                masks = masks[...,0]
            else:
                masks = masks[...,0]
        elif masks.ndim==3:
            if masks.shape[-1]<5:
                masks = masks[np.newaxis,:,:,0]
        elif masks.ndim<3:
            masks = masks[np.newaxis,:,:]
        return masks, self.outlines

    def overlay_transparent(self, bg_img, img_to_overlay_t, maskson=True, outlineson=True):
        # Extract the alpha mask of the RGBA image, convert to RGB 
        b,g,r,a = cv.split(img_to_overlay_t)
        overlay_color = cv.merge((b,g,r)) # Remove alpha(opecity) and matching shape with background image. From (256,256,4) ->(256,156,3)

        mask = cv.medianBlur(a,5) #Remove noise from the image

        # Mask out the logo from the logo image.
        if outlineson and not maskson:
            img2_fg = cv.bitwise_and(overlay_color,overlay_color)
        else:
            img2_fg = cv.bitwise_and(overlay_color,overlay_color,mask = mask)
        
        # Update the original image with our new ROI
        bg_img = cv.add(bg_img, img2_fg)

        return bg_img
        
    def masks_to_outlines(self, masks):
        """ get outlines of masks as a 0-1 array 
        
        Parameters
        ----------------
        masks: int, 2D or 3D array 
            size [Ly x Lx] or [Lz x Ly x Lx], 0=NO masks; 1,2,...=mask labels
        Returns
        ----------------
        outlines: 2D or 3D array 
            size [Ly x Lx] or [Lz x Ly x Lx], True pixels are outlines
        """
        if masks.ndim > 3 or masks.ndim < 2:
            raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array'%masks.ndim)
        self.outlines = np.zeros(masks.shape, bool)
        
        if masks.ndim==3:
            for i in range(masks.shape[0]):
                self.outlines[i] = self.masks_to_outlines(masks[i])
            return self.outlines
        else:
            slices = find_objects(masks.astype(int))
            for i,si in enumerate(slices):
                if si is not None:
                    sr,sc = si
                    mask = (masks[sr, sc] == (i+1)).astype(np.uint8)
                    contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                    pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T            
                    vr, vc = pvr + sr.start, pvc + sc.start 
                    self.outlines[vr, vc] = 1
            return self.outlines

    def process_img_masks(self, image_path, masks_path, currentZ):
        
        self.parent = self.imread(image_path)
        self.masks = self.imread(masks_path)
        self.parent, self.NZ = self._initialize_image(self.parent)

        self.image_path = image_path
        
        self.Ly, self.Lx = self.parent.shape[1:3]
        self.layerz = 255 * np.ones((self.Ly, self.Lx), 'uint8')
        
        self.currentZ = currentZ
        if self.parent.ndim < 4:
            self.onechan=True
        pass
    
        np.random.seed(42) # make colors stable
        colormap = ((np.random.rand(1000000,3)*0.8+0.1)*255).astype(np.uint8)
        self.selected = 0
        self.opacity = 128
        self.strokes = []
        outlinesOn = False
        outpix = np.zeros((1,self.parent.shape[1],self.parent.shape[0]), np.uint32)
        self.outcolor = [200,200,255,200]
        self.cellpix = np.zeros((1, self.parent.shape[1], self.parent.shape[0]), np.uint32)
        self.masks, self.outlines = self._load_masks(self.masks)
        self.shape = self.masks.shape
        self.masks = self.masks.flatten()
        fastremap.renumber(self.masks, in_place=True)
        self.masks = self.masks.reshape(self.shape)
        self.masks = self.masks.astype(np.uint16) if self.masks.max()<2**16-1 else self.masks.astype(np.uint32)
        self.cellpix = self.masks
        if self.cellpix.ndim == 2:
            self.cellpix = self.cellpix[np.newaxis,:,:]
        print(f'GUI_INFO: {self.masks.max()} masks found')

        ncells = self.cellpix.max()
        colors = colormap[:ncells, :3]
        print('GUI_INFO: creating cellcolors and drawing masks')
        self.cellcolors = np.concatenate((np.array([[255,255,255]]), colors), axis=0).astype(np.uint8)


        if self.outlines is None:
            print('INFO: outline is None')
            outpix = np.zeros_like(self.masks)
            for z in range(self.NZ):
                self.outlines = self.masks_to_outlines(self.masks[z])
                outpix[z] = self.outlines * self.masks[z]
                if z%50==0:
                    print('GUI_INFO: plane %d outlines processed'%z)
        else:
            print('INFO: outline is Not None')
            outpix = self.outlines

        return outpix

    def draw_layer(self, image_path, masks_path, currentZ, masksOn=True, outlinesOn=False):
        if self.image_path != image_path:
            self.outpix = self.process_img_masks(image_path, masks_path, currentZ)
        
        self.image_path = image_path
        self.currentZ = currentZ
        self.layerz = np.zeros((self.shape[2],self.shape[1],4), np.uint8)
        
        if masksOn:
            self.layerz = np.zeros((self.Ly,self.Lx,4), np.uint8)
            self.layerz[...,:3] = self.cellcolors[self.cellpix[self.currentZ],:]
            clclr = self.cellcolors[self.cellpix[self.currentZ],:]
            self.layerz[...,3] = self.opacity * (self.cellpix[self.currentZ]>0).astype(np.uint8)
            if self.selected>0:
                self.layerz[self.cellpix[self.currentZ]==self.selected] = np.array([255,255,255, self.opacity])
            cZ = self.currentZ
            stroke_z = np.array([s[0][0] for s in self.strokes])
            inZ = np.nonzero(stroke_z == cZ)[0]
            if len(inZ) > 0:
                for i in inZ:
                    stroke = np.array(self.strokes[i])
                    self.layerz[stroke[:,1], stroke[:,2]] = np.array([255,0,255,100])
                    
        else:
            self.layerz[...,3] = 0

        if outlinesOn:
            if not masksOn:
                self.layerz = np.zeros((self.Ly,self.Lx,4), np.uint8)
            self.layerz[self.outpix[self.currentZ]>0] = np.array(self.outcolor).astype(np.uint8)
        
        img = cv.imread(image_path)
        final_img = self.overlay_transparent(img, self.layerz, maskson=masksOn, outlineson=outlinesOn)

        return self.layerz

overlay = mask_overlay()

if __name__ == "__main__":
    print('mask_overlay_func.py')
