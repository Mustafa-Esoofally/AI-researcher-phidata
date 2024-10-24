import json
from typing import Optional
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import os
import requests
from functools import lru_cache

# Load environment variables from .env file
load_dotenv()

# Access an environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from assistants import (
    SearchTerms,
    search_term_generator,
    arxiv_search_assistant,
    exa_search_assistant,
    research_editor,
    arxiv_toolkit,
)  # type: ignore


st.set_page_config(
    page_title="Research Workflow",
    page_icon=":orange_heart:",
)
st.title("AI Research Workflow")
st.markdown("##### :orange_heart: built by [phidata](https://github.com/phidatahq/phidata)")


def pubmed_search_assistant(search_terms):
    # Example function to search PubMed
    base_url = "https://api.ncbi.nlm.nih.gov/lit/ctxp/v1/pubmed/"
    headers = {"Content-Type": "application/json"}
    results = []

    for term in search_terms:
        response = requests.get(f"{base_url}?term={term}", headers=headers)
        if response.status_code == 200:
            results.append(response.json())
        else:
            print(f"Failed to fetch data for term: {term}")

    return results


@lru_cache(maxsize=32)
def cached_search_terms(topic, num_terms):
    # Function to generate search terms with caching
    return search_term_generator.run(json.dumps({"topic": topic, "num_terms": num_terms}))


def main() -> None:
    # Get topic for report
    input_topic = st.sidebar.text_input(
        ":female-scientist: Enter a topic",
        value="Language Agent Tree Search",
    )
    # Button to generate report
    generate_report = st.sidebar.button("Generate Report")
    if generate_report:
        st.session_state["topic"] = input_topic

    # Checkboxes for search
    st.sidebar.markdown("## Assistants")
    search_exa = st.sidebar.checkbox("Exa Search", value=True)
    search_arxiv = st.sidebar.checkbox("ArXiv Search", value=False)
    search_pubmed = st.sidebar.checkbox("PubMed Search", value=False)
    search_google_scholar = st.sidebar.checkbox("Google Scholar Search", disabled=True)  # noqa
    use_cache = st.sidebar.toggle("Use Cache", value=False, disabled=True)  # noqa
    num_search_terms = st.sidebar.number_input(
        "Number of Search Terms", value=1, min_value=1, max_value=3, help="This will increase latency."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Trending Topics")
    if st.sidebar.button("Language Agent Tree Search"):
        st.session_state["topic"] = "Language Agent Tree Search"

    if st.sidebar.button("AI in Healthcare"):
        st.session_state["topic"] = "AI in Healthcare"

    if st.sidebar.button("Acute respiratory distress syndrome"):
        st.session_state["topic"] = "Acute respiratory distress syndrome"

    if st.sidebar.button("Chromatic Homotopy Theory"):
        st.session_state["topic"] = "Chromatic Homotopy Theory"

    if "topic" in st.session_state:
        report_topic = st.session_state["topic"]

        # Use cached function for generating search terms
        search_terms: Optional[SearchTerms] = cached_search_terms(report_topic, num_search_terms)

        if not search_terms:
            st.write("Sorry report generation failed. Please try again.")
            return

        exa_content: Optional[str] = None
        arxiv_content: Optional[str] = None

        if search_exa:
            with st.status("Searching Exa", expanded=True) as status:
                with st.container():
                    exa_container = st.empty()
                    try:
                        exa_search_results = exa_search_assistant.run(search_terms.model_dump_json(indent=4))
                        if isinstance(exa_search_results, str):
                            raise ValueError("Unexpected string response from exa_search_assistant")
                        if exa_search_results and len(exa_search_results.results) > 0:
                            exa_content = exa_search_results.model_dump_json(indent=4)
                            exa_container.json(exa_search_results.results)
                            status.update(label="Exa Search Complete", state="complete", expanded=False)
                    except Exception as e:
                        st.error(f"An error occurred during Exa search: {e}")
                        status.update(label="Exa Search Failed", state="error", expanded=True)
                        exa_content = None

        if search_arxiv:
            with st.status("Searching ArXiv (this takes a while)", expanded=True) as status:
                with st.container():
                    arxiv_container = st.empty()
                    arxiv_search_results = arxiv_search_assistant.run(search_terms.model_dump_json(indent=4))
                    if arxiv_search_results and arxiv_search_results.results:
                        arxiv_container.json([result.model_dump() for result in arxiv_search_results.results])
                status.update(label="ArXiv Search Complete", state="complete", expanded=False)

            if arxiv_search_results and arxiv_search_results.results:
                paper_summaries = []
                for result in arxiv_search_results.results:
                    summary = {
                        "ID": result.id,
                        "Title": result.title,
                        "Summary": result.summary,
                    }
                    paper_summaries.append(summary)

                if paper_summaries:
                    with st.status("Displaying ArXiv Paper Summaries", expanded=True) as status:
                        with st.container():
                            st.subheader("ArXiv Paper Summaries")
                            df = pd.DataFrame(paper_summaries)
                            st.dataframe(df, use_container_width=True)
                        status.update(label="ArXiv Paper Summaries Displayed", state="complete", expanded=False)

                    arxiv_paper_ids = [summary["ID"] for summary in paper_summaries]
                    if arxiv_paper_ids:
                        with st.status("Reading ArXiv Papers", expanded=True) as status:
                            with st.container():
                                arxiv_content = arxiv_toolkit.read_arxiv_papers(arxiv_paper_ids, pages_to_read=2)
                                st.write(f"Read {len(arxiv_paper_ids)} ArXiv papers")
                            status.update(label="Reading ArXiv Papers Complete", state="complete", expanded=False)

        if search_pubmed:
            with st.status("Searching PubMed", expanded=True) as status:
                pubmed_results = pubmed_search_assistant(search_terms)
                if pubmed_results:
                    st.json(pubmed_results)
                    status.update(label="PubMed Search Complete", state="complete", expanded=False)
                else:
                    st.error("PubMed search failed.")
                    status.update(label="PubMed Search Failed", state="error", expanded=True)

        report_input = ""
        report_input += f"# Topic: {report_topic}\n\n"
        report_input += "## Search Terms\n\n"
        report_input += f"{search_terms}\n\n"

        if arxiv_content:
            report_input += "## ArXiv Papers\n\n"
            report_input += "<arxiv_papers>\n\n"
            report_input += f"{arxiv_content}\n\n"
            report_input += "</arxiv_papers>\n\n"

        if exa_content:
            report_input += "## Web Search Content from Exa\n\n"
            report_input += "<exa_content>\n\n"
            report_input += f"{exa_content}\n\n"
            report_input += "</exa_content>\n\n"

        # New sections for the report
        report_input += "## Summary of Findings\n\n"
        report_input += "Summarize the key findings from the research here.\n\n"

        report_input += "## Key Insights\n\n"
        report_input += "Highlight the most important insights derived from the data.\n\n"

        report_input += "## Conclusion\n\n"
        report_input += "Provide a conclusion based on the research findings.\n\n"

        # Only generate the report if we have content
        if arxiv_content or exa_content:
            with st.spinner("Generating Report"):
                final_report = ""
                final_report_container = st.empty()
                for delta in research_editor.run(report_input):
                    final_report += delta  # type: ignore
                    final_report_container.markdown(final_report)

            # Add an expander to show the list of sources
            with st.expander("List of Sources"):
                st.markdown("### Sources")
                
                # Display ArXiv sources if available
                if arxiv_content:
                    st.markdown("#### ArXiv Papers")
                    arxiv_sources = json.loads(arxiv_content)
                    print("\n\narxiv_content\n\n")
                    print(arxiv_content)
                    for paper in arxiv_sources:
                        st.markdown(f"- **Title**: {paper['title']}")
                        st.markdown(f"  - **Authors**: {', '.join(paper['authors'])}")
                        st.markdown(f"  - **Link**: [PDF]({paper['pdf_url']})")

                # Display Exa sources if available
                if exa_content:
                    try:
                        # Attempt to parse the Exa content as JSON
                        exa_data = json.loads(exa_content)
                        
                        # Access the 'results' key which contains the list of articles
                        exa_sources = exa_data.get("results", [])
                        
                        # Check if the parsed content is a list
                        if isinstance(exa_sources, list):
                            for result in exa_sources:
                                if isinstance(result, dict):
                                    st.markdown(f"- **Title**: {result.get('title', 'No title available')}")
                                    st.markdown(f"  - **Summary**: {result.get('summary', 'No summary available')}")
                                    st.markdown(f"  - **Link**: {result.get('links', ['No URL available'])[0]}")
                                else:
                                    st.error("Unexpected data format in Exa results: Expected a dictionary.")
                        else:
                            st.error("Unexpected data format in Exa results: Expected a list.")
                            # Debugging: Print the raw data to understand its structure
                            st.text(f"Raw Exa content: {exa_content}")
                    except json.JSONDecodeError:
                        st.error("Failed to parse Exa content as JSON.")
        else:
            st.error(
                "Report generation cancelled due to search failure. Please try again or select another search option."
            )

    st.sidebar.markdown("---")
    if st.sidebar.button("Restart"):
        st.rerun()


main()
