# ShopSmart — Hybrid Recommendation System

A hybrid e-commerce recommendation engine combining content-based filtering (TF-IDF + cosine similarity) and collaborative filtering (SVD matrix factorization) with a Tkinter GUI.

## Quick Start

pip install -r requirements.txt
python gui_app.py

## Project Structure

- recommender.py - Core hybrid engine (ContentBasedFilter + CollaborativeFilter + HybridRecommender)
- gui_app.py - Tkinter desktop GUI
- requirements.txt - Python dependencies
- data/products.csv - 50 products across 5 categories
- data/ratings.csv - ~180 ratings from 20 users

## Hybrid Scoring

hybrid_score = alpha * CF_score + (1 - alpha) * CB_score

Default alpha = 0.55. Cold-start users with fewer than 5 ratings get alpha reduced by 0.3 automatically, shifting weight toward content-based signals.

## Dependencies

numpy, pandas, scikit-learn, tkinter (built-in)

## Assignment

University of the Cumberlands — MSAI
Westin Pals, 2025
