"""
gui_app.py
----------
Tkinter GUI for the Hybrid E-Commerce Recommendation System.

Provides a graphical interface allowing users to:
  - Select a user account
  - Filter recommendations by product category
  - Adjust the content vs. collaborative filtering blend weight
  - View personalized recommendations in a sortable table
  - Browse the user's purchase/rating history

Run:
    python gui_app.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from recommender import HybridRecommender


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
BG       = "#1e1e2e"
SIDEBAR  = "#2a2a3e"
CARD     = "#313151"
ACCENT   = "#7c6ee0"
ACCENT2  = "#56cfe1"
TEXT     = "#cdd6f4"
MUTED    = "#888aaa"
SUCCESS  = "#a6e3a1"
WARNING  = "#f9e2af"
WHITE    = "#ffffff"


class RecommenderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ShopSmart — AI Recommendation Engine")
        self.geometry("1100x700")
        self.configure(bg=BG)
        self.resizable(True, True)

        self.engine = HybridRecommender(alpha=0.55)
        self._build_ui()
        self._load_engine_async()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ---- Header ----
        header = tk.Frame(self, bg=ACCENT, height=56)
        header.pack(fill=tk.X, side=tk.TOP)
        tk.Label(
            header, text="ShopSmart  ·  AI Product Recommendations",
            font=("Helvetica", 15, "bold"), bg=ACCENT, fg=WHITE, pady=14
        ).pack(side=tk.LEFT, padx=20)
        self.status_lbl = tk.Label(
            header, text="Loading model…", font=("Helvetica", 10),
            bg=ACCENT, fg=WARNING
        )
        self.status_lbl.pack(side=tk.RIGHT, padx=20)

        # ---- Main layout ----
        main = tk.Frame(self, bg=BG)
        main.pack(fill=tk.BOTH, expand=True)

        self._build_sidebar(main)
        self._build_content(main)

    def _build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=SIDEBAR, width=270)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        sidebar.pack_propagate(False)

        def section(text):
            tk.Label(sidebar, text=text, font=("Helvetica", 9, "bold"),
                     bg=SIDEBAR, fg=MUTED).pack(anchor=tk.W, padx=18, pady=(16, 2))

        # User selection
        section("USER ACCOUNT")
        self.user_var = tk.StringVar()
        self.user_combo = ttk.Combobox(
            sidebar, textvariable=self.user_var, state="disabled",
            font=("Helvetica", 11), width=22
        )
        self.user_combo.pack(padx=18, pady=4, fill=tk.X)

        # Category filter
        section("CATEGORY FILTER")
        self.cat_var = tk.StringVar(value="All")
        self.cat_combo = ttk.Combobox(
            sidebar, textvariable=self.cat_var, state="disabled",
            font=("Helvetica", 11), width=22
        )
        self.cat_combo.pack(padx=18, pady=4, fill=tk.X)

        # Alpha slider
        section("CF BLEND WEIGHT")
        self.alpha_var = tk.DoubleVar(value=0.55)
        alpha_frame = tk.Frame(sidebar, bg=SIDEBAR)
        alpha_frame.pack(padx=18, fill=tk.X)
        self.alpha_slider = tk.Scale(
            alpha_frame, from_=0.0, to=1.0, resolution=0.05,
            orient=tk.HORIZONTAL, variable=self.alpha_var,
            bg=SIDEBAR, fg=TEXT, highlightthickness=0,
            troughcolor=CARD, activebackground=ACCENT,
            length=210, font=("Helvetica", 9)
        )
        self.alpha_slider.pack()
        self.alpha_lbl = tk.Label(
            alpha_frame,
            text="0 = Content-only  ·  1 = Collaborative-only",
            font=("Helvetica", 8), bg=SIDEBAR, fg=MUTED
        )
        self.alpha_lbl.pack()

        # Number of results
        section("NUMBER OF RESULTS")
        self.topn_var = tk.IntVar(value=8)
        topn_frame = tk.Frame(sidebar, bg=SIDEBAR)
        topn_frame.pack(padx=18, fill=tk.X)
        for n in [5, 8, 10, 15]:
            tk.Radiobutton(
                topn_frame, text=str(n), variable=self.topn_var, value=n,
                bg=SIDEBAR, fg=TEXT, selectcolor=CARD,
                activebackground=SIDEBAR, font=("Helvetica", 10)
            ).pack(side=tk.LEFT)

        # Recommend button
        tk.Frame(sidebar, bg=SIDEBAR, height=20).pack()
        self.rec_btn = tk.Button(
            sidebar, text="  Get Recommendations  ",
            font=("Helvetica", 12, "bold"), bg=ACCENT, fg=WHITE,
            activebackground=ACCENT2, activeforeground=WHITE,
            relief=tk.FLAT, cursor="hand2", command=self._run_recommendations,
            state=tk.DISABLED, pady=10
        )
        self.rec_btn.pack(padx=18, fill=tk.X)

        # History button
        tk.Frame(sidebar, bg=SIDEBAR, height=8).pack()
        self.hist_btn = tk.Button(
            sidebar, text="View Rating History",
            font=("Helvetica", 10), bg=CARD, fg=TEXT,
            activebackground=ACCENT, activeforeground=WHITE,
            relief=tk.FLAT, cursor="hand2", command=self._show_history,
            state=tk.DISABLED
        )
        self.hist_btn.pack(padx=18, fill=tk.X, pady=4)

        # Info panel
        tk.Frame(sidebar, bg=SIDEBAR).pack(expand=True)
        info = tk.Frame(sidebar, bg=CARD, padx=12, pady=10)
        info.pack(padx=12, pady=12, fill=tk.X)
        tk.Label(info, text="About the Engine", font=("Helvetica", 9, "bold"),
                 bg=CARD, fg=ACCENT2).pack(anchor=tk.W)
        tk.Label(
            info,
            text=(
                "Hybrid model blending:\n"
                "• Content-based (TF-IDF)\n"
                "• Collaborative (SVD)\n"
                "Cold-start aware — adjusts\n"
                "automatically for sparse users."
            ),
            font=("Helvetica", 8), bg=CARD, fg=MUTED, justify=tk.LEFT
        ).pack(anchor=tk.W)

    def _build_content(self, parent):
        content = tk.Frame(parent, bg=BG)
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=16, pady=12)

        # Tabs
        self.notebook = ttk.Notebook(content)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # --- Recommendations tab ---
        rec_frame = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(rec_frame, text="  Recommendations  ")

        self.rec_label = tk.Label(
            rec_frame,
            text="Select a user and click 'Get Recommendations'",
            font=("Helvetica", 12), bg=BG, fg=MUTED
        )
        self.rec_label.pack(pady=20)

        cols = ("Rank", "Product", "Category", "Price", "CB Score", "CF Score", "Hybrid", "Confidence")
        self.rec_tree = ttk.Treeview(rec_frame, columns=cols, show="headings", height=16)
        widths = (45, 280, 110, 65, 75, 75, 75, 80)
        for col, w in zip(cols, widths):
            self.rec_tree.heading(col, text=col,
                                  command=lambda c=col: self._sort_tree(self.rec_tree, c, False))
            self.rec_tree.column(col, width=w, anchor=tk.CENTER if col != "Product" else tk.W)

        scroll_y = ttk.Scrollbar(rec_frame, orient=tk.VERTICAL, command=self.rec_tree.yview)
        self.rec_tree.configure(yscrollcommand=scroll_y.set)
        self.rec_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=(0, 8))
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y, pady=(0, 8))

        # --- History tab ---
        hist_frame = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(hist_frame, text="  Rating History  ")

        h_cols = ("Product", "Category", "Rating")
        self.hist_tree = ttk.Treeview(hist_frame, columns=h_cols, show="headings", height=18)
        for col in h_cols:
            self.hist_tree.heading(col, text=col)
        self.hist_tree.column("Product",  width=350)
        self.hist_tree.column("Category", width=130)
        self.hist_tree.column("Rating",   width=80, anchor=tk.CENTER)

        h_scroll = ttk.Scrollbar(hist_frame, orient=tk.VERTICAL, command=self.hist_tree.yview)
        self.hist_tree.configure(yscrollcommand=h_scroll.set)
        self.hist_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=8, padx=(0, 0))
        h_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=8)

        # Apply custom styles
        self._apply_styles()

    def _apply_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                         background=CARD, fieldbackground=CARD,
                         foreground=TEXT, rowheight=28,
                         font=("Helvetica", 10))
        style.configure("Treeview.Heading",
                         background=SIDEBAR, foreground=ACCENT2,
                         font=("Helvetica", 10, "bold"))
        style.map("Treeview", background=[("selected", ACCENT)])
        style.configure("TNotebook", background=BG)
        style.configure("TNotebook.Tab", background=SIDEBAR, foreground=MUTED,
                        padding=[12, 6], font=("Helvetica", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", CARD)],
                  foreground=[("selected", WHITE)])

    # ------------------------------------------------------------------
    # Engine Loading (async so UI stays responsive)
    # ------------------------------------------------------------------

    def _load_engine_async(self):
        def _load():
            self.engine.load()
            self.after(0, self._on_engine_loaded)
        threading.Thread(target=_load, daemon=True).start()

    def _on_engine_loaded(self):
        users = self.engine.get_users()
        cats  = self.engine.get_categories()

        self.user_combo["values"] = users
        self.user_combo["state"]  = "readonly"
        self.user_var.set(users[0] if users else "")

        self.cat_combo["values"] = cats
        self.cat_combo["state"]  = "readonly"
        self.cat_var.set("All")

        self.rec_btn["state"]  = tk.NORMAL
        self.hist_btn["state"] = tk.NORMAL
        self.status_lbl.configure(text="Model ready ✓", fg=SUCCESS)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _run_recommendations(self):
        user_id = self.user_var.get()
        if not user_id:
            messagebox.showwarning("No User", "Please select a user.")
            return

        self.rec_btn["state"] = tk.DISABLED
        self.status_lbl.configure(text="Computing…", fg=WARNING)

        def _compute():
            result = self.engine.recommend(
                user_id=user_id,
                top_n=self.topn_var.get(),
                category_filter=self.cat_var.get(),
                alpha_override=round(self.alpha_var.get(), 2),
            )
            self.after(0, lambda: self._display_recommendations(result, user_id))

        threading.Thread(target=_compute, daemon=True).start()

    def _display_recommendations(self, df, user_id):
        for row in self.rec_tree.get_children():
            self.rec_tree.delete(row)

        if df.empty:
            self.rec_label.configure(
                text=f"No recommendations found for {user_id} with current filters."
            )
        else:
            self.rec_label.configure(
                text=f"Top {len(df)} recommendations for {user_id}  "
                     f"(α = {round(self.alpha_var.get(), 2)})"
            )
            for i, row in df.iterrows():
                tag = "high" if row["confidence"] == "High" else "med"
                self.rec_tree.insert(
                    "", tk.END, tags=(tag,),
                    values=(
                        i + 1,
                        row["name"],
                        row["category"],
                        f"${row['price']:.2f}",
                        f"{row['cb_score']:.3f}",
                        f"{row['cf_score']:.3f}",
                        f"{row['hybrid_score']:.3f}",
                        row["confidence"],
                    )
                )
            self.rec_tree.tag_configure("high", foreground=SUCCESS)
            self.rec_tree.tag_configure("med",  foreground=WARNING)

        self.rec_btn["state"] = tk.NORMAL
        self.status_lbl.configure(text="Done ✓", fg=SUCCESS)
        self.notebook.select(0)

    def _show_history(self):
        user_id = self.user_var.get()
        if not user_id:
            return
        history = self.engine.get_rated_products(user_id)
        for row in self.hist_tree.get_children():
            self.hist_tree.delete(row)
        for _, r in history.iterrows():
            stars = "★" * int(r["rating"]) + "☆" * (5 - int(r["rating"]))
            self.hist_tree.insert("", tk.END, values=(r["name"], r["category"], stars))
        self.notebook.select(1)

    def _sort_tree(self, tree, col, reverse):
        rows = [(tree.set(k, col), k) for k in tree.get_children("")]
        try:
            rows.sort(key=lambda x: float(x[0].replace("$", "")), reverse=reverse)
        except ValueError:
            rows.sort(key=lambda x: x[0], reverse=reverse)
        for i, (_, k) in enumerate(rows):
            tree.move(k, "", i)
        tree.heading(col, command=lambda: self._sort_tree(tree, col, not reverse))


if __name__ == "__main__":
    app = RecommenderApp()
    app.mainloop()
