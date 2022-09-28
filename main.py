from predictor import get_predicted_heatmaps, regression_voting, AttrDict
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

labels = AttrDict()
labels.update({
    'S': 1,
    'Na': 2,
    'Or': 3,
    'Po': 4,
    'A': 5,
    'B': 6,
    'Me': 8,
    'Go': 10
})


def getLandmarksCoord(img, h=800, w=640):
    heatmaps, img = get_predicted_heatmaps(img)
    NetOutput = regression_voting(heatmaps, 41).to("cpu")
    output = dict()
    i = 0
    coords = np.array(NetOutput[0].cpu())
    for coord in coords:
        x, y = int(coord[0] * (h - 1)), int(coord[1] * (w - 1))
        output[i] = (x, y)
        i += 1
    return output, heatmaps, img


def lineCoeff(A, B):
    return A[1] - B[1], B[0] - A[0]


def angleFromLines(L1, L2):
    return np.degrees(np.arctan((L2[0] * L1[1] - L1[0] * L2[1]) / (L1[0] * L2[0] + L1[1] * L2[1])))


def getAngles(points_coord):
    output = {
        'SNA': 0,
        'SNB': 0,
        'ANB': 0,
        'SN-MP': 0,
        'FH-MP': 0
    }
    # initializing variables with 2D cephalometric points coordinates
    A, B, S, Na, Or, Po, Me, Go = (np.asarray((points_coord[labels.A - 1][0], points_coord[labels.A - 1][1])),
                                   np.asarray((points_coord[labels.B - 1][0], points_coord[labels.B - 1][1])),
                                   np.asarray((points_coord[labels.S - 1][0], points_coord[labels.S - 1][1])),
                                   np.asarray((points_coord[labels.Na - 1][0], points_coord[labels.Na - 1][1])),
                                   np.asarray((points_coord[labels.Or - 1][0], points_coord[labels.Or - 1][1])),
                                   np.asarray((points_coord[labels.Po - 1][0], points_coord[labels.Po - 1][1])),
                                   np.asarray((points_coord[labels.Me - 1][0], points_coord[labels.Me - 1][1])),
                                   np.asarray((points_coord[labels.Go - 1][0], points_coord[labels.Go - 1][1]))
                                   )
    NS, NA, NB = S - Na, A - Na, B - Na
    output['SNA'] = round(np.degrees(np.arccos(np.dot(NA, NS) / (np.linalg.norm(NA) * np.linalg.norm(NS)))), 2)
    output['SNB'] = round(np.degrees(np.arccos(np.dot(NB, NS) / (np.linalg.norm(NB) * np.linalg.norm(NS)))), 2)
    output['ANB'] = round(output['SNA'] - output['SNB'], 2)
    # For SN-MP and FH-MP
    SN, MP, FH = lineCoeff(S, Na), lineCoeff(Go, Me), lineCoeff(Po, Or)
    output['SN-MP'] = round(abs(angleFromLines(SN, MP)), 2)
    output['FH-MP'] = round(abs(angleFromLines(FH, MP)), 2)
    return output


def angleInterpretation(angles: dict) -> dict:
    output = {
        'Maxilla_to_cranial_base': None,
        'Mandible_to_cranial_base': None,
        'Maxilla_to_Mandible': None,
        'Mandibular_plane': None,
    }
    SNA, SNB, ANB, FMA, SN_MP = angles['SNA'], angles['SNB'], angles['ANB'], angles['FH-MP'], angles['SN-MP']

    # SNA interpretation
    if SNA > 85.0:
        output['Maxilla_to_cranial_base'] = 'prognathic maxilla'
    elif SNA < 79.0:
        output['Maxilla_to_cranial_base'] = 'retrognathic maxilla'
    else:
        output['Maxilla_to_cranial_base'] = 'Normal'

    # SNB interpretation
    if SNB > 82.0:
        output['Mandible_to_cranial_base'] = 'prognathic mandible'
    elif SNB < 76.0:
        output['Mandible_to_cranial_base'] = 'retrognathic mandible'
    else:
        output['Mandible_to_cranial_base'] = 'Normal'

    # ANB interpretation
    if ANB > 5.0:
        output['Maxilla_to_Mandible'] = 'class II'
    elif ANB < 1.0:
        output['Maxilla_to_Mandible'] = 'class III'
    else:
        output['Maxilla_to_Mandible'] = 'Normal'

    # FMA and SN-MP interpretation
    if FMA > 29.0 or SN_MP > 36.0:
        output['Mandibular_plane'] = 'Hyperdivergent'
    elif FMA < 21.0 or SN_MP < 28:
        output['Mandibular_plane'] = 'Hypodivergent'
    else:
        output['Mandibular_plane'] = 'Normodivergent'

    return output


# starting the streamlit app
tab1, tab2 = st.tabs(["Landmarks Heatmaps", "Cephalometric diagnosis"])
st.title("Welcome to the CephaAI Platform")
st.markdown(
    "The cephAI is an cephalometric diagnostics platform, we're deploying here a model cited in this paper \"Cephalometric Landmark Detection by AttentiveFeature Pyramid Fusion and Regression-Voting\"")
st.write(
    "the link to paper [Cephalometric Landmark Detection by AttentiveFeature Pyramid Fusion and Regression-Voting](https://paperswithcode.com/paper/cephalometric-landmark-detection-by) ")
# sidebar
st.sidebar.title("Start testing a profile x-ray image")
uploaded_file = st.sidebar.file_uploader("Choose a profile x-ray image ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image=np.array(image)
    st.sidebar.image(image, caption='Uploaded X-ray image.', use_column_width=True)
    if st.sidebar.button("Get diagnostics"):
        with st.spinner("Processing..."):
            coord, heatmaps ,imm= getLandmarksCoord(image)
            Angles = getAngles(coord)
            interpretations = angleInterpretation(Angles)
        st.sidebar.success("Analysis done! you can show the results")
        with tab1:
            st.header("Cephalometric landmarks heatmaps")
            st.caption("The figue above contain 19 heatmaps, showing the probabilties of regions where each landmark can be located and it's the output of fusionVGG19 network")
            st.caption("Those heatmaps are passed to a regression voting fuction to predict the X,Y normalized coordinates for each landmark")
            fig = plt.figure(figsize=(8, 8))
            columns = 4
            rows = 5
            xr=heatmaps[0][:, 0:19, :].data
            xr=xr.reshape(1, 19, 800,640)
            xr = np.array(xr.cpu())
            for i in range(1, columns * rows - 1):
                img = xr[0][i]
                fig.add_subplot(rows, columns, i)
                plt.imshow(img, cmap='gray', interpolation='nearest')
            st.pyplot(fig)
        with tab2:
            st.header("Cephalometric Diagnosis")
            st.title("Angles values")
            AnglesDf=pd.DataFrame.from_dict(Angles,orient="index")
            st.table(AnglesDf)
            st.title("Cephalometric data interpretation")
            InterpretationDf=pd.DataFrame.from_dict(interpretations,orient="index")
            st.table(InterpretationDf)
