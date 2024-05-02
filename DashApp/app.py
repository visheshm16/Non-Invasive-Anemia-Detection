from dash import Dash, html, dcc, callback, Output, Input, State
import numpy as np
import pandas as pd
import cv2
import base64
from anemia_detect import seg_detect


app = Dash(__name__)
app.css.config.serve_locally = True

app.layout = html.Div(
    [
        html.Div(
            [
                html.H2(
                    children="Anemia Detection Using Conjunctiva-Palor Image",
                    id="page-heading",
                )
            ]
        ),
        html.Div(
            [
                html.H4(
                    children="Upload Image in the field below 👇", id="upload-heading"
                ),
                dcc.Upload(
                    html.Button(["Select Image 📷"]),
                    id="upload-image",
                    multiple=False,
                    accept="image/*",
                ),
            ]
        ),
        html.Div(id="output-containder"),
    ]
)


@callback(
    Output(component_id="output-containder", component_property="children"),
    Input("upload-image", "contents"),
)
def update_output(content):
    if content:
        if not content.startswith("data:"):
            raise ValueError("Invalid base64 string format.")

        print(f"Image received type : {type(content)}")
        header, data = content.split(",", 1)
        print(f"Header : {header}")
        im_format = header.split("/", 1)[1].split(";")[0]
        print(f"Format : {im_format}")

        decoded_content = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
        if im_format == "jpeg":
            decoded_content = cv2.imdecode(decoded_content, cv2.IMREAD_COLOR)
        elif im_format == "png":
            decoded_content = cv2.imdecode(decoded_content, cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError("Unsupported image format: {}".format(im_format))

        print(f"Decoded image type : {type(decoded_content)}")

        rim, diagnosis, conf = seg_detect(decoded_content)

        if not rim.any():
            el = html.Div(
                [
                    html.H5(children="Image uploaded 👇", id="confirmation"),
                    html.Img(src=content, style={"height": "10vh"}),
                    html.P(
                        children=f"""{conf:.2f}% chance that you are {dianosis}.
                This is only a suggestive diagnosis as this model completely relies on redness analysis of the conjunctiva.
                Hence, it is recommended to get your bloodwork done to know your actual hemoglobin levels.""",
                        id="output-message",
                    ),
                ]
            )
        else:
            retval, buffer = cv2.imencode(".jpeg", rim)
            base64_string = base64.b64encode(buffer).decode("utf-8")
            print(header + "," + base64_string[:50])

            el = html.Div(
                [
                    html.H5(children="Image uploaded 👇", id="confirmation"),
                    html.Div(children=[html.Img(src=content, id='original', style={"height": "20vh"})]),
                    html.Div(
                        [
                            html.Img(
                                src=header + "," + base64_string,
                                id='segmented',
                                style={"height": "20vh"},
                            )
                        ]
                    ),
                    html.P(
                        children=f"{conf:.2f}% chance that you are {diagnosis}.",
                        id="output-message",
                    ),
                    html.P(
                        children="""This is only a suggestive diagnosis as this model completely relies on redness analysis of the conjunctiva.
                Hence, it is recommended to <b>get your bloodwork done to know your actual hemoglobin levels</b>.""",
                        id='user-warning',
                    ),
                ]
            )
        return el


if __name__ == "__main__":
    app.run_server(debug=True)
