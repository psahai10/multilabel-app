import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from glob import glob
import json
import ast
import numpy as np
from typing import List
from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from pandas.api.types import is_numeric_dtype

from utils import (
    save_session,
    load_session,
    load_label_json,
    update_label_json,
    update_session_file,
    download_image
)

#st.set_page_config(layout="wide")

HOME_DIRECTORY = ".streamlit"
if os.path.exists(HOME_DIRECTORY) is False:
    os.makedirs(HOME_DIRECTORY)


################# Helper functions ###########################################
@st.cache(allow_output_mutation=True)
def get_state():
    return {}


@st.cache(allow_output_mutation=True)
def get_image_csv(csv_file_path):
    img_df = pd.read_csv(csv_file_path)
    if "img_path" not in img_df:
        st.error("'img_path' doesn't exist in the table")
    if "label" in img_df:
        img_df = img_df.drop(["label"], axis=1)
    return img_df


@st.cache(allow_output_mutation=True)
def dataframe_join_image_label(img_df, label_df):
    img_df = img_df.join(label_df, on="img_path", how="left")
    img_df.loc[img_df["label"].isna(), "label"] = ""
    return img_df

def display_session_setup():
    col1, col2, col3 = st.columns([2.5, 2, 2])
    #username = st.session_state["username"]
    #username = 'default'
    username = col1.text_input("Please Enter your user name", "default_user")
    col1.write("Current User: " + username)
    session_file = os.path.join(HOME_DIRECTORY, username + "multilabel_image_session")

    if os.path.exists(session_file) and username != "":
        col2.write(" ")
        col2.write(" ")
        # col2.write("Load sa")
        load_existing_session = col2.button(
            "Load Saved Session", key="load_existing_session"
        )
        # col2.text("or")
        # diff_session_file = col2.file_uploader(
        #     "Choose a session file on disk", key="diff_session_file"
        # )
    # else:
    #     load_existing_session = False
    #     col1.write("No session File Found")
    #     diff_session_file = col2.file_uploader(
    #         "Choose a session file on disk", key="diff_session_file"
    #     )

    def start_new_session_fn():
        clean_session()
        # st.session_state["username"] = username
        save_session(session_file, st.session_state)

    col3.write(" ")
    col3.write(" ")
#     col2.write(" ")
#     col2.write(" ")
    start_new_session = col3.button(
        "Start New Session", on_click=start_new_session_fn, key="start_new_session"
    )
    #return username, load_existing_session, diff_session_file, start_new_session
    return username, load_existing_session, None, start_new_session


def update_session_profile(session_dict):
    if "label_path" in session_dict:
        st.session_state["label_path"] = session_dict["label_path"]
    if "image_folder" in session_dict:
        st.session_state["image_folder"] = session_dict["image_folder"]
    if "image_csv_file" in session_dict:
        st.session_state["image_csv_file"] = session_dict["image_csv_file"]
    if "username" in session_dict:
        st.session_state["username"] = session_dict["username"]
    if "filter1_values" in session_dict:
        st.session_state["filter1_values"] = tuple(session_dict["filter1_values"])
    if "multiple_labels" in session_dict:
        st.session_state["multiple_labels"] = session_dict["multiple_labels"]


def update_multiple_labels(multiple_labels):
    st.session_state["multiple_labels"] = multiple_labels
    update_session_file()


def clean_session():
    st.session_state["label_path"] = ""
    st.session_state["image_folder"] = ""
    if "filter1_values" in st.session_state:
        del st.session_state["filter1_values"]
    if "multiple_labels" in st.session_state:
        del st.session_state["multiple_labels"]


def label_category(label_str):
    if len(label_str) == 0:
        return "Unlabeled"
    labels = label_str.split(",")
    if len(labels) == 1:
        return "Single Label"
    else:
        return "Multi Label"


def extract_all_labels(label_str_list):
    all_labels = []
    for label_str in label_str_list:
        if len(label_str) > 0:
            labels = label_str.split(",")
            all_labels.extend(labels)
    return all_labels


def extract_multilabels(label_str_list):
    all_labels = []
    for label_str in label_str_list:
        if len(label_str) > 0:
            labels = label_str.split(",")
            if len(labels) > 1:
                all_labels.append(",".join(sorted(labels)))
    return all_labels


def step1_setup_session():
    (username,
     load_existing_session,
     diff_session_file,
     start_new_session,
    ) = display_session_setup()
    session_file = os.path.join(HOME_DIRECTORY, username + "multilabel_image_session")
    st.session_state["session_file"] = session_file
    st.session_state["username"] = username
    if load_existing_session and os.path.exists(session_file):
        data = load_session(session_file)
        # if "username" in data:
        #     del data["username"]
        update_session_profile(data)
    if diff_session_file is not None:
        data = json.load(diff_session_file)
        # if "username" in data:
        #     del data["username"]
        update_session_profile(data)
        save_session(session_file, st.session_state)
    #return username


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def step2_setup_image_label_path():

    st.subheader("1. Set Image Path")
    image_path_col1 , image_path_col2, image_path_col3 = st.columns([2, 1, 1])

    path = 'dataset/labels/'
    filename_list = sorted(glob(os.path.join(path, "*.csv")))

    image_csv_file = image_path_col1.selectbox(
    "Please input the path to csv file with 'img_path' column to the images", 
    tuple(filename_list), on_change=update_session_file)

    try:
        file_name = image_csv_file.split('.')[0].split('/')[-1]
        username = st.session_state["username"]
        json_file = username + '_' + file_name + '.json'
        label_path = 'dataset/results/' + json_file
    except Exception as e:
        print("Error setting file in where to save labels")
    if not label_path or not label_path.endswith('.json'):
        st.subheader("2. Set the Label Path")
        label_path = st.text_input(
            "Please input the path to save the label (json format)",
            key="label_path",
            on_change=update_session_file,
        )

    list_of_cols = ["img_path", "img_name", "color_name", "color_code", "color_family", "ligthness", "chroma", "nine_color_group"]
    img_df = pd.DataFrame(columns=list_of_cols)
    label_df = pd.DataFrame(columns=["label"])
    label_dict = {}
    derived_labels = ""

    if image_csv_file != "":
        if os.path.exists(image_csv_file) is False:
            st.error(image_csv_file + " doesn't exist")
        else:
            img_df = get_image_csv(image_csv_file)
    else:
        st.error("image path is not specified")

    if label_path == "":
        st.error("label_path is not specified, please set the path to save the labels")
    else:
        if os.path.exists(label_path):
            label_dict = load_label_json(label_path)
            label_df = pd.Series(label_dict).to_frame(name="label")
            json_labels = []
            for _, val in label_dict.items():
                json_labels.extend(val.split(","))
            derived_labels = ",".join(sorted(np.unique(json_labels)))

    return img_df, label_df, label_dict, label_path, derived_labels


def step3_display_and_label(derived_labels, state, img_df, label_dict, label_path):
    st.subheader("2. Label the images one by one")
    (
        img_width_col,
        img_height_col,
        label_input_buff,
        save_multilabels_to_session_col,
    ) = st.columns([0.8, 0.8, 2.2, 1])
    target_width = img_width_col.number_input(
        "Image width", min_value=10, max_value=1000, value=300
    )
    target_height = img_height_col.number_input(
        "height", min_value=10, max_value=1000, value=300
    )
    union_labels = []
    if "multiple_labels" in st.session_state:
        union_labels = ",".join(
            sorted(
                list(
                    set(derived_labels).union(
                        st.session_state["multiple_labels"].split(",")
                    )
                )
            )
        )
        multiple_labels = label_input_buff.text_input(
            "Multi-Label Options (seperate by ,)",
            value=st.session_state["multiple_labels"],
        )
    else:
        if len(derived_labels) == 0:
            multiple_labels = label_input_buff.text_input(
                "Multi-Label Options (seperate by ,)", value="Evergreen,Fundamental,Fundamental+,Edge,Seasonal,Sports,Style,Uncertain,OutofScope"
            )
        else:
            multiple_labels = label_input_buff.text_input(
                "Multi-Label Options (seperate by ,)", value=derived_labels
            )
    save_multilabels_to_session_col.button(
        "Save Multi-label Options to Session File",
        on_click=update_multiple_labels,
        args=(multiple_labels,),
    )

    target_labels = [_.strip() for _ in multiple_labels.split(",")]


    def reset_session():
        clean_session()
        session_file = st.session_state["session_file"]
        save_session(session_file, st.session_state)
        # os.remove(session_file)
        st.experimental_rerun()
        return


    st.button("Start New Session?", on_click=reset_session)

    col1, col2, col3 = st.columns([1.3, 2.0, 2.0])
    with col1:

        if "current_row" not in state or state["current_row"] is None:
            state["current_row"] = 0
        if "current_row" in state and state["current_row"] is not None:
            if state["current_row"] < 0:
                state["current_row"] = len(img_df) - 1
            elif state["current_row"] >= len(img_df):
                state["current_row"] = 0

        prev_click = st.button("previous")
        next_click = st.button("next")
        
        current_row = state["current_row"]
        if prev_click is True:
            current_row -= 1
            if current_row < 0:
                current_row = len(img_df) - 1
        if next_click is True:
            current_row += 1
            if current_row >= len(img_df):
                current_row = 0
        current_row = st.selectbox(
            "Select an image for labeling", img_df.index, current_row
        )

        if current_row is not None:
            state["current_row"] = current_row

    with col2:
        asset_id = img_df.loc[current_row, "img_path"]
        image = download_image(asset_id)
        image = image.resize((target_width, target_height))
        image_name = img_df.loc[current_row, "img_name"]
        st.image(image, caption=image_name)

    current_labels = img_df.loc[current_row, "label"].split(",")
    selected_labels = col3.multiselect(
        "Current Label", options=target_labels + [""], default=current_labels
    )
    username = st.session_state["username"]
    key = username+'-'+asset_id
    label_dict[key] = ",".join(sorted([_ for _ in selected_labels if _ != ""]))
    col3.button(
        "Update Label",
        on_click=update_label_json,
        args=(label_path, label_dict),
    )
    pass


def step4_color_table(state, img_df):

    def get_list_of_dic(a,b,c,d,e,f):
        list_of_colors = [a,b,c,d,e,f]

        return list_of_colors
 
    def str_to_dict(string: str):
        # Color codes we haven't found have a color for
        unapproved_list = ['0BE', '84E', '6AP', '78E', '90Z', '9MQ', '91L', 
                        '4GQ', '4PL', '90A', '93D', '81E', '2CW', '93I', 
                        '97J', '9RY', '24S', '3MM', '47Q', '01D', '9SN', 
                        '99Z', '9SQ', '9GQ', '00N', '91B', '04G', '49F', 
                        '2CS', '3MY', '6CI', '91M', '4QL', '6JI', '91N', 
                        '92A', '92K', '0AX', '05S', '72Y', '9GM', '10C', 
                        '11N', '50N', '4QR', '05R', '43U', '90T', '54F', 
                        '76D', '9SS', '30C', '90I', '4GN', '3EV', '4HB', 
                        None, np.nan, 'None', 'nan', 'NAN', 'NaN']
        if string not in unapproved_list:
            return ast.literal_eval(string)
        else:
            return None

    def get_colors_cols(list_of_cols: List[dict]):
        names_of_cols = ['PRIMARY', 'SECONDARY', 'TERTIARY', 'QUATERNARY', 'LOGO', 'LOGO_ACCENT']
        temp = {}
        assert len(list_of_cols)==len(names_of_cols)
        for idx in range(len(list_of_cols)):
            v = str_to_dict(list_of_cols[idx])
            if v is not None:
                temp[names_of_cols[idx]] = {
                    'color_name': v['color_name'], 
                    'color_code': v['color_code'], 
                    'color_family': v['color_family'], 
                    'ligthness' : v['ligthness'],
                    'chroma' : v['chroma'],
                    'nine_color_group': v['nine_color_group']
                }
        return  pd.DataFrame.from_dict(temp, orient='index')

    current_row = state["current_row"]
    p = img_df.loc[current_row, "PRIMARY"]
    s = img_df.loc[current_row, "SECONDARY"]
    t = img_df.loc[current_row, "TERTIARY"]
    q = img_df.loc[current_row, "QUATERNARY"]
    l = img_df.loc[current_row, "LOGO"]
    la = img_df.loc[current_row, "LOGO_ACCENT"]
    list_of_colors = get_list_of_dic(p,s,t,q,l,la)
    df = get_colors_cols(list_of_colors)
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.write('No associated color were found')

def step5_download_json():
    from pathlib import Path
    path = 'dataset/results/'
    json_files = sorted(glob(os.path.join(path, "*.json")))
    if json_files:
        st.subheader("3. Download labels below.")
        #st.markdown("**:red[Download labels below]**")
        st.text("Please make sure to first save the files using the 'Save Multi-label Options to Sesson File' button above.")
        for file in json_files:
            st.download_button(
                label='Download {}.json'.format(file.split('/')[-1].split('.')[0]),
                data=Path(file).read_text(),
                file_name='{}.json'.format(file.split('/')[-1].split('.')[0]),
                mime="application/json",
            )

def step6_label_stats(img_df):
    #### Step 3
    path = 'dataset/results/'
    json_files = sorted(glob(os.path.join(path, "*.json")))
    if json_files:
        st.subheader("4. Label Stats")
    else:
        st.subheader("3. Label Stats")
    
    # try:
    #     plt.style.use('seaborn')
    # except:
    #     pass

    fig, axes = plt.subplots(2, 2)
    fig.set_figheight(8)
    fig.set_figwidth(12)
    img_df["label_category"] = img_df["label"].apply(lambda x: label_category(x))
    img_df.groupby("label_category").count()["img_path"].plot(
        kind="barh", ax=axes[0][0]
    )
    axes[0][0].set_title("Label Category")

    all_labels = extract_all_labels(img_df["label"].values)
    w = Counter(all_labels)
    axes[0][1].barh(list(w.keys()), list(w.values()))
    axes[0][1].set_title("All Labels")

    single_labels = extract_all_labels(
        img_df[img_df["label_category"] == "Single Label"]["label"].values
    )
    w2 = Counter(single_labels)
    axes[1][0].barh(list(w2.keys()), list(w2.values()))
    axes[1][0].set_title("Single Labels")

    multi_labels = extract_multilabels(
        img_df[img_df["label_category"] == "Multi Label"]["label"].values
    )
    w3 = Counter(multi_labels)
    axes[1][1].barh(list(w3.keys()), list(w3.values()))
    axes[1][1].set_title("Multi Labels")

    st.pyplot(fig)


def main():

    ### Instruction
    st.header("Multi-label Image Classification Labeling Tool")
    st.markdown(
        """
        This is an image classification labeling APP created with Streamlit. It supports multi-labels for the same image.
        Please follow the instructions to login, session setup, load, label images and save your labels
    """
    )

    # if check_password():

    # st.write(st.session_state)

    ## User Session Handle
    session_ready = False
    if "session_file" in st.session_state and os.path.exists(
        st.session_state["session_file"]
    ):
        session_ready = True
        data = load_session(st.session_state["session_file"])
        # if "username" in data:
        #     del data["username"]
        update_session_profile(data)
        st.write(
            "Current session are loaded from %s"
            % (st.session_state["session_file"])
        )
    else:
        st.subheader("Set Up Session")
        step1_setup_session()

    if session_ready:
        (
            img_df,
            label_df,
            label_dict,
            label_path,
            derived_labels,
        ) = step2_setup_image_label_path()

        img_df = dataframe_join_image_label(img_df, label_df)

        state = get_state()
    
        if len(img_df) == 0:
            st.error("No images are selected for labels")
        else:
            step3_display_and_label(
                derived_labels, state, img_df, label_dict, label_path
            )
            step4_color_table(state, img_df)
            step5_download_json()
            step6_label_stats(deepcopy(img_df))


if __name__ == "__main__":
    main()