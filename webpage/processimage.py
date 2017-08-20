#!/usr/bin/env python

from PIL import Image
import base64
import cgi
import cPickle
import io
import json
import uuid


# whether we should save the various steps of the image processing to disk
# very useful when debugging to ensure the images look valid.
SAVE_IMAGES = True


def process_image(serialised_image_data):
    """
    Takes a base64 encoded image:
      * decodes it
      * centers it
      * scales it
      * binary thresholds it
      * applies it to the neural network
    """
    image_id = str(uuid.uuid4())

    original_image = deserialise_image(serialised_image_data)

    if SAVE_IMAGES:
        original_image.save("%s.original.png" % (image_id,))

    # center the image, by pasting it on to an ideal background size
    width, height = original_image.size
    if height >= width:
        centered_image = Image.new("RGB", (height, height), (255, 255, 255))
        offset_x = (height - width) / 2
        centered_image.paste(original_image, (offset_x, 0))
    else:
        centered_image = Image.new("RGB", (width, width), (255, 255, 255))
        offset_y = (width - height) / 2
        centered_image.paste(original_image, (0, offset_y))

    if SAVE_IMAGES:
        centered_image.save("%s.centered.png" % (image_id,))

    # resize image to 16 * 16
    resized_image = centered_image.resize((16, 16))

    if SAVE_IMAGES:
        resized_image.save("%s.resized.png" % (image_id,))

    # binary threshold image (along original threshold)
    bw_image = resized_image \
        .convert("L") \
        .point(lambda pixel: 0 if pixel < 128 else 255, "1")

    if SAVE_IMAGES:
        bw_image.save("%s.bw.png" % (image_id,))

    # prepare for neural network
    binary_image_data = map(lambda x: 0 if x > 127 else 1, bw_image.getdata())

    return binary_image_data


def deserialise_image(data):
    """
    Turns the provided base64 encoded data into an PIL Image.
    """
    if "data:image" in data:
        data = data[data.find(",") + 1:]

    return Image.open(io.BytesIO(base64.urlsafe_b64decode(data)))


def predict_image(serialised_image):
    """
    Decodes the given image data and use it to generate a
    prediction of the number it indicates.
    """
    network_input = process_image(serialised_image)

    with open("trained_network.pickle", "r") as fin:
        trained_network = cPickle.load(fin)

    trained_network.forward(network_input)
    output_layer = trained_network.output_layer

    return output_layer.index(max(output_layer))


def respond_to_request():
    """
    The main body of the CGI script.
    """
    form_data = cgi.FieldStorage()

    # we only want to respond to requests that send an image
    if "img" not in form_data.keys():
        print "Status: 400 Bad Request"
        print

        exit()

    try:
        # get the image data
        serialised_image_data = str(form_data["img"].value).strip()
        response = predict_image(serialised_image_data)

        # send headers
        print "Content-type: aplication/json"
        print

        # send the result
        print json.dumps({
            "output": response,
        })

    except Exception:
        print "Status: 500 Internal Server Error"
        print


if __name__ == "__main__":
    respond_to_request()
