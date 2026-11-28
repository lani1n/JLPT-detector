from nlp_logic import analyze_JLPT
import streamlit as st


OUTPUT_CSV = "data/vocabulary.csv"

st.set_page_config(page_title="JLPT Detector", layout="wide")
st.title("JLPT Detector")

# Optional controls
with st.sidebar:
    st.header("Settings")
    level = st.selectbox(
        "JLPT level (this level and above)",
        ["N5", "N4", "N3", "N2", "N1"],
        index=2  # N3
    )
    max_words = st.slider("Maximum words", 5, 30, 20)
    include_n0 = st.checkbox("Include above N1 (N0)", value=False)

# Top: input on the left, translation on the right
left, right = st.columns(2)

with left:
    text = st.text_area("Japanese", height=300)
    clicked = st.button("Translate", type="primary")

if clicked:
    # Clear old results so they never appear beside new input
    st.session_state.pop("result", None)

    if not text.strip():
        st.warning("Please enter some Japanese text.")
    else:
        try:
            with st.spinner("Analyzing..."):
                st.session_state["result"] = analyze_JLPT(
                    text,
                    OUTPUT_CSV,
                    level=level,
                    max_words=max_words,
                    no_N0_words=not include_n0
                )
        except Exception as error:
            st.error(f"Something went wrong: {error}")

result = st.session_state.get("result")
translation = result[0] if result else ""

with right:
    st.text_area("Thai", value=translation, height=300)

# Bottom: vocabulary table
if result:
    df = result[1]

    st.subheader(f"Vocabulary ({len(df)})")

    if df.empty:
        st.info("No words found at the selected levels.")
    else:
        st.dataframe(df, hide_index=True)
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="vocabulary.csv",
            mime="text/csv"
        )