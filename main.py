import numpy

from cnn_architectures.architectures.models.double_unet.double_unet import DoubleUNet

import cv2

if __name__ == "__main__":
    r"""
    model = DoubleUNet((256, 256, 3))
    model.build()
    model.load_weight(r"C:\Users\Usuario\OneDrive - Universitat de les Illes Balears\UIB\Investigacio\Proyectos I+D\Redscar\Server\Code\redscar_server\redscar\redscar_api\model\image_analysis\ml_models\wound_segmentation\dun3_epochs=100_lr=3e-5_res=0.h5")

    image = cv2.imread(r"C:\Users\Usuario\OneDrive - Universitat de les Illes Balears\UIB\Investigacio\Proyectos I+D\Redscar\Server\Code\redscar_server\staticfiles\temp_results\rPERcMLTUK_11012023\original_image.png")

    prediction_mask_bool, prediction_mask_int, prediction_colormask = model.predict_binary(image=image, binary_threshold=0.5, prediction_index=1)
    """
    mask = cv2.imread(r"C:\Users\Usuario\Desktop\wound_segmentation\djnIcpMtGUmA2PAxIYTDpdD9y_mask.png", cv2.IMREAD_GRAYSCALE)
    I = cv2.imread(r"C:\Users\Usuario\Desktop\wound_segmentation\djnIcpMtGUmA2PAxIYTDpdD9y_gsc.png")

    mask = cv2.resize(mask, (I.shape[1], I.shape[0]))
    prediction_mask_bool = numpy.zeros_like(mask, dtype=bool)
    prediction_mask_bool[mask == 255] = True

    I[:, :, 0][prediction_mask_bool] = 0
    I[:, :, 1][prediction_mask_bool] = 255
    I[:, :, 2][prediction_mask_bool] = 0

    cv2.imwrite(r"C:\Users\Usuario\Desktop\wound_segmentation\djnIcpMtGUmA2PAxIYTDpdD9y_gsc.png", I)