class Dataset():
    def __init__(self):
        i = 0
            
    
    
    
    
    def get_patches(self, spectrum, size=[3, 3]):
        pad = [int((s - 1) / 2) for s in size]
        if size[0] % 2 == 1 and size[1] % 2 == 1:
            spectrum_ = np.pad(spectrum, ((pad[0], pad[0]), (pad[1], pad[1]), (0, 0)))
        elif size[0] % 2 == 1 and size[1] % 2 == 0:
            spectrum_ = np.pad(spectrum, ((pad[0], pad[0]), (pad[1], pad[1] + 1), (0, 0)))
        elif size[0] % 2 == 0 and size[1] % 2 == 1:
            spectrum_ = np.pad(spectrum, ((pad[0], pad[0] + 1), (pad[1], pad[1]), (0, 0)))
        elif size[0] % 2 == 0 and size[1] % 2 == 0:
            spectrum_ = np.pad(spectrum, ((pad[0], pad[0] + 1), (pad[1], pad[1] + 1), (0, 0)))

        patches = image.extract_patches_2d(spectrum_, tuple(size))  
        patches = np.reshape(patches, (spectrum.shape[0], spectrum.shape[1], size[0], size[1], patches.shape[-1]))

        return patches
    
    def save_patches_to_npz():
        
        i=0
        spectrum_3d = None
        mask_3d = None
        _indexes = None

        _data_ = np.load('utils/for_patches.npz', allow_pickle=True)


        for mask, spectrum, healthy_indexes, ill_indexes, y_, name in zip(_data_['masks'], _data_['spectrums'],_data_['healthy_indexes_all'], _data_['ill_indexes_all'], _data_['y_all'], _data_['names']):
            patches = get_patches(spectrum, size=[3, 3])
            print('Shape of patches:', patches.shape)

            if i == 0:
                spectrum_3d = patches.copy()
                mask_3d = mask.copy()
                _indexes = healthy_indexes | ill_indexes

            healthy_spectrum = patches[healthy_indexes]
            ill_spectrum = patches[ill_indexes]
            X_3d = np.array(list(healthy_spectrum) + list(ill_spectrum))

            np.savez(os.path.join('data_3d_npz', name), X = X_3d, y = y_)

            i+=1


        X, y = get_X_y('data_3d_npz/*.npz')
        X, y = shuffle(X, y)
        train_X_3d, train_y_3d, test_X_3d, test_y_3d = split(X, y)

        print('train_X_3d and train_y_3d shapes: ', train_X_3d.shape, train_y_3d.shape)

        assert train_X.shape[0] == train_X_3d.shape[0]

        np.savez('utils/draw_3d', mask=mask_3d[..., ::-1], spectrum=spectrum_3d, indexes=_indexes)