#!/usr/bin/env python3
"""
BBBC013 Interactive CMPO Phenotype Explorer

Interactive Streamlit application for exploring condition-specific CMPO phenotypes
from the BBBC013 protein translocation dataset analysis.
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from collections import defaultdict, Counter
import networkx as nx

# Page config
st.set_page_config(
    page_title="BBBC013 CMPO Phenotype Explorer", 
    page_icon="🧬", 
    layout="wide"
)

@st.cache_data
def load_analysis_results():
    """Load the BBBC013 phenotype analysis results."""
    results_file = Path("bbbc013_phenotype_results.json")
    if not results_file.exists():
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)

@st.cache_data
def prepare_data(results):
    """Prepare data for visualization."""
    if not results:
        return None, None, None
    
    # Extract condition summaries and CMPO terms
    conditions = []
    all_cmpo_terms = []
    
    for well_id, result in results.items():
        condition = result['condition']
        condition_key = f"{condition['treatment']}_{condition['concentration']}uM"
        
        for cmpo_term in result['cmpo_terms']:
            all_cmpo_terms.append({
                'well_id': well_id,
                'condition': condition_key,
                'treatment': condition['treatment'],
                'concentration': condition['concentration'],
                'term_name': cmpo_term.get('term_name', 'Unknown'),
                'confidence': cmpo_term.get('confidence', 0),
                'stage': cmpo_term.get('stage', 'unknown'),
                'cmpo_id': cmpo_term.get('CMPO_ID', 'Unknown')
            })
        
        conditions.append({
            'well_id': well_id,
            'condition': condition_key,
            'treatment': condition['treatment'],
            'concentration': condition['concentration'],
            'replicate': condition.get('replicate', 1),
            'condition_type': condition.get('condition_type', 'Unknown')
        })
    
    df_conditions = pd.DataFrame(conditions)
    df_cmpo = pd.DataFrame(all_cmpo_terms)
    
    # Aggregate CMPO terms by condition
    cmpo_summary = df_cmpo.groupby(['condition', 'term_name']).agg({
        'confidence': ['mean', 'max', 'count', 'std']
    }).round(3)
    cmpo_summary.columns = ['mean_confidence', 'max_confidence', 'frequency', 'std_confidence']
    cmpo_summary = cmpo_summary.reset_index()
    
    return df_conditions, df_cmpo, cmpo_summary

def create_network_visualization(df_cmpo, selected_conditions):
    """Create interactive network visualization of CMPO terms and conditions."""
    # Filter data
    filtered_df = df_cmpo[df_cmpo['condition'].isin(selected_conditions)]
    
    if filtered_df.empty:
        return None
    
    # Create network data
    G = nx.Graph()
    
    # Add condition nodes
    for condition in selected_conditions:
        G.add_node(condition, node_type='condition', size=20)
    
    # Add CMPO term nodes and edges
    term_weights = defaultdict(float)
    condition_term_edges = []
    
    for _, row in filtered_df.iterrows():
        term = row['term_name']
        condition = row['condition']
        confidence = row['confidence']
        
        # Add term node
        if term not in G.nodes():
            G.add_node(term, node_type='term', size=10)
        
        # Track edge weights
        edge_key = (condition, term)
        term_weights[edge_key] += confidence
        
        condition_term_edges.append({
            'condition': condition,
            'term': term,
            'confidence': confidence
        })
    
    # Add edges with aggregated weights
    for (condition, term), weight in term_weights.items():
        G.add_edge(condition, term, weight=weight)
    
    return G, condition_term_edges

def create_confidence_heatmap(cmpo_summary, selected_conditions, top_n_terms=15):
    """Create confidence heatmap for selected conditions."""
    # Filter for selected conditions
    filtered_summary = cmpo_summary[cmpo_summary['condition'].isin(selected_conditions)]
    
    if filtered_summary.empty:
        return None
    
    # Get top terms by frequency across all selected conditions
    term_frequency = filtered_summary.groupby('term_name')['frequency'].sum().sort_values(ascending=False)
    top_terms = term_frequency.head(top_n_terms).index.tolist()
    
    # Create pivot table
    heatmap_data = filtered_summary[filtered_summary['term_name'].isin(top_terms)].pivot(
        index='condition', columns='term_name', values='mean_confidence'
    ).fillna(0)
    
    # Create plotly heatmap
    fig = px.imshow(
        heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale='YlOrRd',
        aspect='auto',
        title=f"Top {top_n_terms} CMPO Terms by Condition"
    )
    
    fig.update_xaxes(side="bottom")
    fig.update_layout(
        xaxis_title="CMPO Terms",
        yaxis_title="Experimental Conditions",
        height=400
    )
    
    return fig

def create_dose_response_plot(df_cmpo, selected_term):
    """Create dose-response plot for a specific CMPO term."""
    if not selected_term:
        return None
    
    # Filter for selected term
    term_data = df_cmpo[df_cmpo['term_name'] == selected_term].copy()
    
    if term_data.empty:
        return None
    
    # Aggregate by treatment and concentration
    dose_response = term_data.groupby(['treatment', 'concentration'])['confidence'].agg(['mean', 'std']).reset_index()
    
    fig = go.Figure()
    
    treatments = dose_response['treatment'].unique()
    colors = px.colors.qualitative.Set1
    
    for i, treatment in enumerate(treatments):
        treatment_data = dose_response[dose_response['treatment'] == treatment]
        
        fig.add_trace(go.Scatter(
            x=treatment_data['concentration'],
            y=treatment_data['mean'],
            error_y=dict(type='data', array=treatment_data['std']),
            mode='markers+lines',
            name=treatment,
            line=dict(color=colors[i % len(colors)]),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title=f"Dose-Response for: {selected_term}",
        xaxis_title="Concentration (µM)",
        yaxis_title="Mean Confidence",
        xaxis_type="log",
        height=400
    )
    
    return fig

def main():
    """Main Streamlit application."""
    st.title("🧬 BBBC013 CMPO Phenotype Explorer")
    st.markdown("**Interactive analysis of condition-specific phenotypes from protein translocation experiments**")
    
    # Load data
    results = load_analysis_results()
    
    if not results:
        st.error("❌ Analysis results not found. Please run `bbbc013_phenotype_analysis.py` first.")
        st.info("💡 Run: `python bbbc013_phenotype_analysis.py` to generate the required data.")
        return
    
    df_conditions, df_cmpo, cmpo_summary = prepare_data(results)
    
    if df_cmpo.empty:
        st.warning("⚠️ No CMPO terms found in the analysis results.")
        return
    
    # Sidebar controls
    st.sidebar.header("🎛️ Analysis Controls")
    
    # Condition selection
    available_conditions = sorted(df_conditions['condition'].unique())
    selected_conditions = st.sidebar.multiselect(
        "Select Experimental Conditions:",
        available_conditions,
        default=available_conditions[:4]  # Default to first 4
    )
    
    if not selected_conditions:
        st.warning("Please select at least one experimental condition.")
        return
    
    # Analysis options
    analysis_type = st.sidebar.selectbox(
        "Analysis Type:",
        ["Overview", "Condition Comparison", "Dose-Response Analysis", "Term Networks"]
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("📊 Quick Stats")
        
        filtered_cmpo = df_cmpo[df_cmpo['condition'].isin(selected_conditions)]
        
        st.metric("Conditions Selected", len(selected_conditions))
        st.metric("Total CMPO Terms", len(filtered_cmpo['term_name'].unique()))
        st.metric("Total Observations", len(filtered_cmpo))
        
        # Top terms
        top_terms = filtered_cmpo['term_name'].value_counts().head(5)
        st.write("**🏆 Top CMPO Terms:**")
        for term, count in top_terms.items():
            st.write(f"• {term[:40]}{'...' if len(term) > 40 else ''}: {count}")
    
    with col1:
        if analysis_type == "Overview":
            st.subheader("📈 Analysis Overview")
            
            # Confidence distribution
            fig_dist = px.histogram(
                filtered_cmpo, 
                x='confidence', 
                color='treatment',
                nbins=20,
                title="Confidence Score Distribution by Treatment"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Stage distribution
            stage_counts = filtered_cmpo['stage'].value_counts()
            fig_stage = px.pie(
                values=stage_counts.values,
                names=stage_counts.index,
                title="CMPO Terms by Pipeline Stage"
            )
            st.plotly_chart(fig_stage, use_container_width=True)
        
        elif analysis_type == "Condition Comparison":
            st.subheader("🔍 Condition Comparison")
            
            # Heatmap
            heatmap_fig = create_confidence_heatmap(cmpo_summary, selected_conditions)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Differential analysis
            st.write("**🎯 Condition-Specific Terms:**")
            
            for condition in selected_conditions:
                condition_terms = cmpo_summary[cmpo_summary['condition'] == condition]
                top_condition_terms = condition_terms.nlargest(3, 'mean_confidence')
                
                st.write(f"**{condition}:**")
                for _, term in top_condition_terms.iterrows():
                    st.write(f"  • {term['term_name']}: {term['mean_confidence']:.2f} confidence")
        
        elif analysis_type == "Dose-Response Analysis":
            st.subheader("📈 Dose-Response Analysis")
            
            # Term selection for dose-response
            available_terms = sorted(filtered_cmpo['term_name'].unique())
            selected_term = st.selectbox("Select CMPO Term:", available_terms)
            
            if selected_term:
                dose_fig = create_dose_response_plot(filtered_cmpo, selected_term)
                if dose_fig:
                    st.plotly_chart(dose_fig, use_container_width=True)
                
                # Show term details
                term_details = filtered_cmpo[filtered_cmpo['term_name'] == selected_term]
                st.write(f"**📋 Details for: {selected_term}**")
                st.write(f"• Observations: {len(term_details)}")
                st.write(f"• Mean Confidence: {term_details['confidence'].mean():.2f}")
                st.write(f"• Confidence Range: {term_details['confidence'].min():.2f} - {term_details['confidence'].max():.2f}")
        
        elif analysis_type == "Term Networks":
            st.subheader("🌐 CMPO Term Networks")
            
            try:
                G, edges = create_network_visualization(filtered_cmpo, selected_conditions)
                
                if G and len(G.nodes()) > 0:
                    # Network statistics
                    st.write("**📊 Network Statistics:**")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Nodes", len(G.nodes()))
                    with col_b:
                        st.metric("Edges", len(G.edges()))
                    with col_c:
                        st.metric("Density", f"{nx.density(G):.3f}")
                    
                    # Show network summary
                    st.write("**🔗 Network Connections:**")
                    edge_df = pd.DataFrame(edges)
                    if not edge_df.empty:
                        network_summary = edge_df.groupby(['condition', 'term'])['confidence'].mean().reset_index()
                        st.dataframe(network_summary.head(10))
                else:
                    st.warning("No network data available for selected conditions.")
            
            except ImportError:
                st.warning("NetworkX not available for network analysis. Install with: `pip install networkx`")
    
    # Data export
    st.sidebar.subheader("💾 Export Data")
    
    if st.sidebar.button("📊 Download CMPO Summary"):
        csv_data = cmpo_summary[cmpo_summary['condition'].isin(selected_conditions)]
        st.sidebar.download_button(
            label="📁 Download CSV",
            data=csv_data.to_csv(index=False),
            file_name="bbbc013_cmpo_summary.csv",
            mime="text/csv"
        )
    
    # Show raw data
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("📋 Raw CMPO Data")
        st.dataframe(filtered_cmpo)

if __name__ == "__main__":
    main()