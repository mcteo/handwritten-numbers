i#!/usr/bin/env python

from uuid import uuid4
from PIL import Image
from io import BytesIO
import json
import base64
import cPickle
import cgi


SAVE_IMAGES = False


def process_image(serialised_image_data):
    """
    Takes a base64 encoded image:
      * decodes it
      * scales it
      * binary thresholds it
      * applies it to the neural network
    """

    image_id = str(uuid4())

    original_image = deserialise_image(serialised_image_data)
    if SAVE_IMAGES:
        original_image.save(image_id + "-original.png")

    # resize image to 16 * 16
    resized_image = original_image.resize((16, 16))
    # NEAREST, BILINEAR, BICUBIC, ANTIALIAS
    if SAVE_IMAGES:
        resized_image.save(image_id + "-resized.png")

    # binary threshold image (along original threshold)
    bw_image = resized_image \
        .convert("L") \
        .point(lambda pixel: 0 if pixel < 128 else 255, "1")
    if SAVE_IMAGES:
        bw_image.save(image_id + "-resized-bw.png")

    # prepare for neural network
    binary_image_data = list(bw_image.getdata())
    binary_image_data = map(lambda x: 0 if x > 127 else 1, binary_image_data)

    return binary_image_data


def deserialise_image(data):
    """
    """
    if "data:image" in data:
        data = data[data.find(",") + 1:]

    return Image.open(BytesIO(base64.b64decode(data)))


if __name__ == "__main__":

    post_data = cgi.FieldStorage()

    # 404
    if "img" not in get_data.keys():
        print "Status: 404 Not Found"
        print

        exit()

    try:
        serialised_image_data = str(get_data["img"].value).strip()
        network_input = process_image(serialised_image_data)

        with open("trained_network.pickle", "r") as fin:
            trained_network = cPickle.load(fin)

        trained_network.forward(network_input)
        output = trained_network.output_layer
        prediction = output.index(max(output))

        print "Content-type: aplication/json"
        print

        print json.dumps({
            "prediction": str(prediction),
            "output": output,           
        })

    except Exception:
        print "Status: 500 Internal Server Error"
        print
