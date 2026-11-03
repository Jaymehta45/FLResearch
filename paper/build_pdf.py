#!/usr/bin/env python3
"""Build a readable multi-page IEEE-style PDF with ReportLab (no blank Chrome PDF)."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OUT = ROOT / "ieee_paper.pdf"


def styles():
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle(
            "TitleIEEE",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=14,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "author": ParagraphStyle(
            "AuthorIEEE",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=11,
            alignment=TA_CENTER,
            spaceAfter=3,
        ),
        "affil": ParagraphStyle(
            "AffilIEEE",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=9,
            alignment=TA_CENTER,
            spaceAfter=14,
            textColor=colors.HexColor("#333333"),
        ),
        "h1": ParagraphStyle(
            "H1IEEE",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=11,
            leading=14,
            spaceBefore=14,
            spaceAfter=6,
            textTransform="uppercase",
        ),
        "h2": ParagraphStyle(
            "H2IEEE",
            parent=base["Heading2"],
            fontName="Times-BoldItalic",
            fontSize=10.5,
            leading=13,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "BodyIEEE",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=10,
            leading=13,
            alignment=TA_JUSTIFY,
            spaceAfter=7,
        ),
        "abs": ParagraphStyle(
            "AbsIEEE",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=9.5,
            leading=12,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "CapIEEE",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=8.5,
            leading=11,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=10,
        ),
        "eq": ParagraphStyle(
            "EqIEEE",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            spaceBefore=6,
            spaceAfter=6,
        ),
        "ref": ParagraphStyle(
            "RefIEEE",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=9,
            leading=11,
            leftIndent=12,
            firstLineIndent=-12,
            spaceAfter=4,
        ),
        "tablecap": ParagraphStyle(
            "TableCap",
            parent=base["Normal"],
            fontName="Times-Bold",
            fontSize=9,
            alignment=TA_CENTER,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "algo": ParagraphStyle(
            "AlgoIEEE",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=9,
            leading=11,
            leftIndent=8,
            spaceAfter=2,
        ),
    }
    return s


def P(text: str, style):
    return Paragraph(text, style)


def fig(story, s, filename: str, caption: str, width=6.3 * inch):
    path = FIG / filename
    if not path.exists():
        story.append(P(f"[Missing figure: {filename}]", s["caption"]))
        return
    img = Image(str(path))
    img._restrictSize(width, 4.2 * inch)
    story.append(KeepTogether([img, P(caption, s["caption"])]))


def bullets(items, s):
    flow = []
    for it in items:
        flow.append(ListItem(P(it, s["body"]), leftIndent=12, bulletColor=colors.black))
    return ListFlowable(flow, bulletType="bullet", start="•", leftIndent=15)


def numbered(items, s):
    flow = []
    for it in items:
        flow.append(ListItem(P(it, s["body"]), leftIndent=12))
    return ListFlowable(flow, bulletType="1", leftIndent=15)


def table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def build():
    s = styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Federated Averaging for Multi-Class Network Intrusion Classification",
        author="Jay Mehta",
    )
    story = []

    story.append(
        P(
            "Federated Averaging for Multi-Class Network Intrusion Classification:<br/>"
            "Theoretical Foundations, Empirical Evaluation, and Resilience to Label Poisoning<br/>"
            "on the CIC–UNSW-NB15 Dataset",
            s["title"],
        )
    )
    story.append(P("Jay Mehta", s["author"]))
    story.append(P("Independent Research / Academic Project", s["affil"]))

    story.append(
        P(
            "<b><i>Abstract—</i></b>"
            "Federated learning (FL) enables multiple institutions to jointly train machine learning "
            "models without exchanging raw data, a property that is especially desirable for network "
            "security analytics where flow telemetry may contain sensitive operational information. "
            "This paper presents a comprehensive theoretical and empirical study of Federated Averaging "
            "(FedAvg) applied to multi-class intrusion / traffic classification on the CIC–UNSW-NB15 "
            "augmented dataset. We formalize the FL objective, the FedAvg update rule, local empirical "
            "risk minimization with multilayer perceptrons (MLPs), and a single-client label-poisoning "
            "threat model. We then conduct a 64-configuration experimental sweep over client counts, "
            "global communication rounds, local epochs, and poison intensities. Empirically, longer "
            "training improves clean accuracy (best clean accuracy ≈ 0.924), while targeted poisoning is "
            "most harmful when a single client dominates training (worst accuracy ≈ 0.856). With multiple "
            "clients, FedAvg dilutes one poisoned shard and accuracy typically remains above 0.90. "
            "Confusion-matrix and one-vs-rest TP/FP/FN/TN analyses show that majority-class targeted "
            "poisoning systematically increases false positives for class 0 and reduces recall for "
            "several minority attack categories—an effect that can be understated by overall accuracy "
            "alone under severe class imbalance. Interactive dashboard: "
            "<link href='https://fl-research.vercel.app'>https://fl-research.vercel.app</link>.",
            s["abs"],
        )
    )
    story.append(
        P(
            "<b><i>Keywords—</i></b>"
            "Federated learning, Federated Averaging (FedAvg), intrusion detection systems, network "
            "traffic classification, data poisoning, label flipping, CIC–UNSW-NB15, multilayer "
            "perceptron, confusion matrix, class imbalance",
            s["abs"],
        )
    )

    # I
    story.append(P("I. INTRODUCTION", s["h1"]))
    story.append(P("A. Motivation", s["h2"]))
    story.append(
        P(
            "Network intrusion detection and traffic classification increasingly rely on supervised "
            "learning over high-dimensional flow features. In operational settings, however, the "
            "organizations that collect such telemetry—enterprises, internet service providers, campus "
            "networks, or industrial control environments—are often unwilling or legally unable to "
            "centralize raw records. Privacy regulations, competitive sensitivity, and breach-risk all "
            "discourage naive data pooling.",
            s["body"],
        )
    )
    story.append(
        P(
            "Federated learning offers an alternative: each participant trains on local data and shares "
            "only model parameters (or gradients), which a coordinating server aggregates into a global "
            "model. For intrusion detection, this paradigm suggests a path toward collaborative detection "
            "quality without collaborative data warehouses.",
            s["body"],
        )
    )

    story.append(P("B. Problem Statement", s["h2"]))
    story.append(
        P(
            "We study the following concrete problem. Given a tabular multi-class dataset with feature "
            "vectors x ∈ ℝ<sup>d</sup> and labels y ∈ {0,…,C−1}, simulate N clients holding disjoint "
            "training shards and learn a shared classifier f<sub>θ</sub> via FedAvg. We then ask:",
            s["body"],
        )
    )
    story.append(
        numbered(
            [
                "How do N, the number of global rounds R, and local epochs E affect test accuracy and F1-score?",
                "How does label poisoning on a single client, with fraction f of labels corrupted, change global performance?",
                "Does poisoning alter <i>which</i> classes are confused, beyond aggregate metrics?",
            ],
            s,
        )
    )

    story.append(P("C. Scope and Simulation Nature", s["h2"]))
    story.append(
        P(
            "All experiments in this repository run on a single machine. “Clients” are logical partitions "
            "of a shuffled training set rather than physically distributed hosts. The simulation preserves "
            "the algorithmic structure of FL (local optimization + server aggregation) while abstracting "
            "away network latency, dropouts, and systems engineering. This is appropriate for controlled "
            "scientific comparison of learning dynamics and poisoning effects, but it is not a full systems "
            "deployment study.",
            s["body"],
        )
    )

    story.append(P("D. Contributions", s["h2"]))
    story.append(
        numbered(
            [
                "A self-contained theoretical presentation of FedAvg for multi-class classification, including the global objective, local surrogate losses, weighted aggregation, and a formal label-poisoning adversary.",
                "An end-to-end reproducible pipeline on CIC–UNSW-NB15 augmented data: cleaning, stratified splitting, scaling, IID sharding, MLP training, and evaluation with confusion matrices and per-class TP/FP/FN/TN.",
                "A 64-run experimental grid quantifying interactions among clients, rounds, local epochs, and poison settings, with best-case clean accuracy near 0.924 and worst-case poisoned accuracy near 0.856.",
                "Evidence that majority-class targeted poisoning induces a “predict class 0 when uncertain” bias that is most severe under single-client training and largely diluted under multi-client FedAvg.",
                "Public interactive visualizations of the full experimental record.",
            ],
            s,
        )
    )

    story.append(P("E. Paper Organization", s["h2"]))
    story.append(
        P(
            "Section II reviews FL, IDS learning, and poisoning. Section III develops the mathematical "
            "framework. Section IV describes the system design. Section V details the experimental protocol. "
            "Sections VI–VII present and interpret results. Section VIII concludes.",
            s["body"],
        )
    )

    # II
    story.append(P("II. BACKGROUND AND RELATED WORK", s["h1"]))
    story.append(P("A. Centralized Learning for Intrusion Detection", s["h2"]))
    story.append(
        P(
            "Classical machine learning for IDS ranges from shallow models (decision trees, SVMs, random "
            "forests) to deep neural networks trained on packet or flow features. Centralized training "
            "assumes a single data lake. While often accurate in benchmarks, this assumption conflicts with "
            "multi-organization collaboration. Moreover, network data are characteristically <i>imbalanced</i>: "
            "benign or majority classes dominate, so accuracy can be misleading unless accompanied by "
            "per-class analysis. Cost-sensitive learning, resampling, and anomaly-detection formulations are "
            "common responses; in this work we retain a standard multi-class MLP and instead emphasize "
            "diagnostic metrics.",
            s["body"],
        )
    )
    story.append(P("B. Federated Learning Basics", s["h2"]))
    story.append(
        P(
            "In cross-silo or cross-device FL, a server coordinates rounds of local training and aggregation. "
            "FedAvg is the canonical algorithm: clients run multiple local SGD/Adam steps and return updated "
            "weights; the server averages them, typically weighted by local dataset sizes. The literature has "
            "since developed methods for statistical heterogeneity (FedProx, SCAFFOLD), communication "
            "compression, personalization, secure aggregation, and robust aggregation against Byzantine "
            "clients. Understanding vanilla FedAvg remains foundational because it is still the default "
            "baseline in many applied FL systems.",
            s["body"],
        )
    )
    story.append(
        P(
            "A central systems insight of FedAvg is that increasing local computation (more local epochs) "
            "can reduce the number of communication rounds needed to reach a target accuracy when client "
            "data are not too heterogeneous. That trade-off is directly relevant to IDS collaborations "
            "spanning administrative domains with limited bandwidth or intermittent connectivity.",
            s["body"],
        )
    )
    story.append(P("C. FL for Security Analytics", s["h2"]))
    story.append(
        P(
            "Prior work has explored FL for IoT and network intrusion detection to keep telemetry local "
            "while improving collective detectors. Our contribution is not a new FL optimizer; rather, we "
            "provide a carefully instrumented FedAvg case study with poisoning analysis and confusion-level "
            "diagnostics on a widely used tabular intrusion dataset, accompanied by public artifacts.",
            s["body"],
        )
    )
    story.append(P("D. Poisoning and Byzantine Behavior", s["h2"]))
    story.append(
        P(
            "Data poisoning injects corrupted training examples to degrade or redirect a learned model. In "
            "FL, a malicious client can poison labels or features, or upload crafted weights (model poisoning "
            "/ backdoor insertion). Defenses include robust aggregators (coordinate-wise median, trimmed mean, "
            "Krum), anomaly detection on updates, and differential privacy. This paper evaluates "
            "<i>undefended</i> FedAvg against a simple but realistic label-flipping adversary on one client, "
            "establishing an empirical baseline for how much “natural dilution” multi-client averaging provides "
            "before specialized defenses are added.",
            s["body"],
        )
    )
    story.append(P("E. UNSW-NB15 Family of Datasets", s["h2"]))
    story.append(
        P(
            "UNSW-NB15 and related CIC-augmented releases provide labeled flow features covering multiple "
            "attack categories alongside normal traffic. We use an augmented CIC–UNSW-NB15 tabular release "
            "with d = 76 numeric features and C = 10 classes. Exact semantic names of minority classes are "
            "less critical for our algorithmic conclusions than the empirical imbalance pattern: one dominant "
            "class and several rare classes with near-zero recall under a simple MLP. This imbalance is "
            "typical of security telemetry and motivates confusion-matrix reporting.",
            s["body"],
        )
    )

    # III Theory
    story.append(P("III. THEORETICAL FRAMEWORK", s["h1"]))
    story.append(P("A. Supervised Multi-Class Risk", s["h2"]))
    story.append(
        P(
            "Let 𝒟 be a distribution over 𝒳 × 𝒴 with 𝒴 = {0,…,C−1}. A parametric model f<sub>θ</sub>: "
            "𝒳 → ℝ<sup>C</sup> produces logits; the predicted label is ŷ = arg max<sub>c</sub> "
            "[f<sub>θ</sub>(x)]<sub>c</sub>. The population risk under cross-entropy is",
            s["body"],
        )
    )
    story.append(P("R(θ) = 𝔼<sub>(x,y)∼𝒟</sub> [ ℓ(f<sub>θ</sub>(x), y) ]     (1)", s["eq"]))
    story.append(
        P(
            "where ℓ(z, y) = − log ( e<sup>z<sub>y</sub></sup> / Σ<sub>c</sub> e<sup>z<sub>c</sub></sup> ). "
            "Given a finite training set S = {(x<sub>i</sub>, y<sub>i</sub>)}<sub>i=1</sub><sup>n</sup>, "
            "the empirical risk is",
            s["body"],
        )
    )
    story.append(
        P(
            "R̂<sub>S</sub>(θ) = (1/n) Σ<sub>i=1</sub><sup>n</sup> ℓ(f<sub>θ</sub>(x<sub>i</sub>), y<sub>i</sub>).     (2)",
            s["eq"],
        )
    )

    story.append(P("B. Federated Partition of the Empirical Risk", s["h2"]))
    story.append(
        P(
            "Partition S into N disjoint client datasets S<sub>k</sub> with n<sub>k</sub> = |S<sub>k</sub>| "
            "and Σ<sub>k</sub> n<sub>k</sub> = n. The global empirical risk decomposes exactly as a mixture "
            "of local empirical risks:",
            s["body"],
        )
    )
    story.append(
        P(
            "R̂<sub>S</sub>(θ) = Σ<sub>k=1</sub><sup>N</sup> (n<sub>k</sub>/n) R̂<sub>S<sub>k</sub></sub>(θ).     (3)",
            s["eq"],
        )
    )
    story.append(
        P(
            "Equation (3) is the conceptual justification for size-weighted FedAvg: averaging models trained "
            "on local risks approximates minimizing a weighted combination of those risks when local updates "
            "remain sufficiently close to the broadcast initialization (limited client drift).",
            s["body"],
        )
    )

    story.append(P("C. Local Optimization", s["h2"]))
    story.append(
        P(
            "In round t, client k receives θ<sup>t</sup> and approximately solves",
            s["body"],
        )
    )
    story.append(
        P(
            "θ<sub>k</sub><sup>t+1</sup> ≈ arg min<sub>θ</sub> R̂<sub>S<sub>k</sub></sub>(θ)  initialized at θ<sup>t</sup>,     (4)",
            s["eq"],
        )
    )
    story.append(
        P(
            "using E epochs of minibatch Adam with learning rate η. Let B be the minibatch size. The number "
            "of local gradient steps per round is roughly E · ⌈n<sub>k</sub>/B⌉. More local steps increase "
            "client drift away from θ<sup>t</sup> but also increase local progress per communication round—"
            "the classic FedAvg trade-off between computation and communication. In our implementation, the "
            "Adam optimizer state is re-initialized each round from the broadcast weights, so momentum does "
            "not persist across rounds.",
            s["body"],
        )
    )

    story.append(P("D. FedAvg Aggregation", s["h2"]))
    story.append(
        P(
            "The server sets",
            s["body"],
        )
    )
    story.append(
        P(
            "θ<sup>t+1</sup> = Σ<sub>k=1</sub><sup>N</sup> w<sub>k</sub> θ<sub>k</sub><sup>t+1</sup>,  "
            "w<sub>k</sub> = n<sub>k</sub> / Σ<sub>j</sub> n<sub>j</sub>.     (5)",
            s["eq"],
        )
    )
    story.append(
        P(
            "If shards are equal-sized, w<sub>k</sub> = 1/N. Aggregation is performed element-wise over all "
            "tensors in the state dictionary (weights and biases of every layer). Because every client shares "
            "the same architecture, corresponding parameters are aligned and averaging is well-defined.",
            s["body"],
        )
    )

    story.append(P("E. Communication Cost Model", s["h2"]))
    story.append(
        P(
            "Let |θ| denote the number of scalar parameters. Each round transmits approximately N|θ| "
            "parameters uplink and N|θ| downlink (broadcast counted per client), i.e., Θ(N|θ|) communication "
            "per round. Total communication over R rounds scales as Θ(R N |θ|). Increasing E can reduce the "
            "R required for a target accuracy in many IID settings, improving communication efficiency—one of "
            "FedAvg’s original motivations. For our compact MLP, |θ| is only tens of thousands of scalars, "
            "making full-model exchanges inexpensive in simulation.",
            s["body"],
        )
    )

    story.append(P("F. MLP Hypothesis Class", s["h2"]))
    story.append(
        P(
            "Our model is a two-hidden-layer MLP:",
            s["body"],
        )
    )
    story.append(
        P(
            "f<sub>θ</sub>(x) = W<sub>3</sub> σ(W<sub>2</sub> σ(W<sub>1</sub>x + b<sub>1</sub>) + b<sub>2</sub>) + b<sub>3</sub>,     (6)",
            s["eq"],
        )
    )
    story.append(
        P(
            "with σ = ReLU, hidden width 128, d = 76, and C = 10. Softmax is implicit in the cross-entropy "
            "loss. ReLU networks can approximate a wide family of piecewise-linear decision regions; in "
            "practice, capacity interacts with class imbalance, often favoring majority-class decision rules "
            "unless rare classes are sufficiently represented or reweighted.",
            s["body"],
        )
    )

    story.append(P("G. Class Imbalance and Metrics", s["h2"]))
    story.append(
        P(
            "Overall accuracy Acc = (1/n<sub>test</sub>) Σ<sub>i</sub> 1[ŷ<sub>i</sub> = y<sub>i</sub>] "
            "is dominated by majority-class correctness under imbalance. We therefore also report precision, "
            "recall, F1, the full confusion matrix M ∈ ℕ<sup>C×C</sup> with "
            "M<sub>ij</sub> = #{i′ : y<sub>i′</sub>=i, ŷ<sub>i′</sub>=j}, and one-vs-rest counts:",
            s["body"],
        )
    )
    story.append(
        P(
            "TP<sub>c</sub>=M<sub>cc</sub>,  FP<sub>c</sub>=Σ<sub>i≠c</sub> M<sub>ic</sub>,  "
            "FN<sub>c</sub>=Σ<sub>j≠c</sub> M<sub>cj</sub>,  "
            "TN<sub>c</sub>=n<sub>test</sub>−TP<sub>c</sub>−FP<sub>c</sub>−FN<sub>c</sub>.     (8)",
            s["eq"],
        )
    )
    story.append(
        P(
            "Per-class recall TP<sub>c</sub>/(TP<sub>c</sub>+FN<sub>c</sub>) is especially informative for "
            "rare attacks. High TN<sub>c</sub> values are expected in multi-class settings because most "
            "examples are not of class c; thus TP/FP/FN trends are usually more actionable than TN magnitude alone.",
            s["body"],
        )
    )

    story.append(P("H. Threat Model: Label Poisoning", s["h2"]))
    story.append(
        P(
            "We consider a dirty-label adversary controlling client 0 only. Features remain untouched; a "
            "fraction f ∈ [0,1] of local labels are replaced according to a mode m:",
            s["body"],
        )
    )
    story.append(
        bullets(
            [
                "<b>Target:</b> y ← y<sub>tgt</sub> (we use y<sub>tgt</sub> = 0, the majority class).",
                "<b>Random-wrong:</b> y ← uniform over 𝒴 ∖ {y<sub>true</sub>}.",
                "<b>Cycle:</b> y ← (y+1) mod C.",
            ],
            s,
        )
    )
    story.append(
        P(
            "The adversary does not modify uploaded weights beyond ordinary training on poisoned labels, "
            "and does not control the test distribution. This models a compromised or negligent data pipeline "
            "at one silo rather than an optimal model-replacement attack.",
            s["body"],
        )
    )

    story.append(P("I. Why Dilution Occurs under FedAvg", s["h2"]))
    story.append(
        P(
            "Suppose equal shard sizes and one poisoned client. Even if θ<sub>0</sub><sup>t+1</sup> is "
            "strongly biased, the aggregate is",
            s["body"],
        )
    )
    story.append(
        P(
            "θ<sup>t+1</sup> = (1/N) θ<sub>0</sub><sup>t+1</sup> + (1/N) Σ<sub>k=1</sub><sup>N−1</sup> "
            "θ<sub>k</sub><sup>t+1</sup>.     (9)",
            s["eq"],
        )
    )
    story.append(
        P(
            "As N grows, the poisoned update’s coefficient 1/N shrinks. This is not a cryptographic defense "
            "and fails if many clients collude or if the adversary mounts stronger model poisoning; but it "
            "explains our empirical observation that multi-client runs remain near clean accuracy under "
            "single-client label flips. Conversely, when N = 1, equation (9) collapses to "
            "θ<sup>t+1</sup> = θ<sub>0</sub><sup>t+1</sup>, so poisoning transfers undiluted.",
            s["body"],
        )
    )

    story.append(P("J. Convergence Intuition (IID Case)", s["h2"]))
    story.append(
        P(
            "When shards are IID draws from the same distribution, local risks are statistically similar. "
            "Local Adam steps from a shared initialization tend to move in related directions, so averaging "
            "is constructive. Formal FedAvg analyses typically assume smoothness/Lipschitz gradients, bounded "
            "stochastic gradient variance, and control of client drift as a function of E and heterogeneity. "
            "Our shards are IID-like by construction (random shuffle then cut), placing us near the favorable "
            "regime. Non-IID partitions would increase drift and could amplify poisoning or reduce clean "
            "accuracy; we leave that to future work.",
            s["body"],
        )
    )
    story.append(
        P(
            "An intuitive view is that FedAvg performs a form of inexact alternating minimization on the "
            "mixture objective (3): each client descends on its term, then the server projects back to a "
            "consensus model by averaging. With small E and IID data, the inexactness is mild; with large E "
            "or heterogeneous data, consensus quality degrades without algorithmic corrections (e.g., proximal terms).",
            s["body"],
        )
    )

    story.append(P("K. On Targeted Majority-Class Poisoning", s["h2"]))
    story.append(
        P(
            "Targeting the majority class is a strategically strong dirty-label choice under imbalance. "
            "Flipping labels to class 0 increases the empirical prior toward 0 inside the poisoned shard. "
            "The resulting local optimum tends to expand the region of feature space assigned to class 0. "
            "After aggregation, especially at N = 1, test points from neighboring classes are more often "
            "absorbed into class 0 predictions. Random-wrong flips, by contrast, inject less coherent bias "
            "and often degrade performance less at equal f in our sweep.",
            s["body"],
        )
    )

    # IV
    story.append(P("IV. SYSTEM DESIGN AND IMPLEMENTATION", s["h1"]))
    story.append(P("A. Data Pipeline", s["h2"]))
    story.append(
        P(
            "Figure 1 summarizes preprocessing. Features and labels are aligned row-wise from Dataset.csv "
            "and Label.csv. Non-finite values are replaced by zeros. A stratified split (default 80% train / "
            "20% test) preserves class proportions. A StandardScaler is fit on training features only, then "
            "applied to train and test. Training indices are shuffled with a fixed seed and split into N "
            "approximately equal shards.",
            s["body"],
        )
    )
    fig(story, s, "fig_pipeline.png", "Fig. 1. End-to-end data pipeline from raw CSVs to client shards and a held-out test set.")
    story.append(
        P(
            "<b>Note on global scaling.</b> Fitting a scaler on pooled training features is a practical "
            "simplification for a single-machine simulation. Strictly decentralized FL might estimate "
            "normalization statistics via secure aggregation or use per-client normalization.",
            s["body"],
        )
    )

    story.append(P("B. Model Architecture", s["h2"]))
    story.append(
        P(
            "Figure 2 shows the shared MLP. Identical architecture across clients is mandatory for "
            "element-wise FedAvg. The network is intentionally compact to enable a large configuration "
            "sweep on CPU.",
            s["body"],
        )
    )
    fig(story, s, "fig_mlp.png", "Fig. 2. Shared MLP classifier used by every client.")

    story.append(P("C. Federated Training Loop", s["h2"]))
    story.append(
        P(
            "Figure 3 and Algorithm 1 describe one round of training. In code, each client deep-copies the "
            "global state dictionary, trains with Adam and cross-entropy, then returns parameters and sample "
            "counts for weighted averaging.",
            s["body"],
        )
    )
    fig(story, s, "fig_fedavg.png", "Fig. 3. One FedAvg round: broadcast, local training, and weighted aggregation.")
    story.append(P("<b>Algorithm 1:</b> FedAvg for intrusion classification (simulation)", s["tablecap"]))
    for line in [
        "1. Initialize global model θ<sup>0</sup>.",
        "2. For round t = 0 … R−1:",
        "&nbsp;&nbsp;&nbsp;a. For each client k = 1…N: set θ<sub>k</sub> ← θ<sup>t</sup>; run E local Adam epochs on S<sub>k</sub>; upload (θ<sub>k</sub>, n<sub>k</sub>).",
        "&nbsp;&nbsp;&nbsp;b. Aggregate θ<sup>t+1</sup> ← Σ<sub>k</sub> (n<sub>k</sub>/Σ<sub>j</sub> n<sub>j</sub>) θ<sub>k</sub>.",
        "&nbsp;&nbsp;&nbsp;c. Evaluate θ<sup>t+1</sup> on the held-out test set.",
        "3. Return θ<sup>R</sup>.",
    ]:
        story.append(P(line, s["algo"]))
    story.append(Spacer(1, 6))

    story.append(P("D. Poisoning Hook", s["h2"]))
    story.append(
        P(
            "Before local training begins, if poisoning is enabled, client 0’s label vector is rewritten "
            "according to Section III-H (Figure 4). Features and all other clients remain clean. The test "
            "set is never poisoned.",
            s["body"],
        )
    )
    fig(story, s, "fig_poison.png", "Fig. 4. Targeted label poisoning applied only to client 0 training labels.")

    story.append(P("E. Software Stack and Configuration", s["h2"]))
    story.append(
        P(
            "The implementation uses PyTorch for the MLP and optimization; pandas/NumPy for I/O; and "
            "scikit-learn for splitting, scaling, confusion matrices, and classification reports. A "
            "configuration dataclass exposes clients, rounds, local epochs, batch size, learning rate, "
            "device, and poison parameters via CLI flags, enabling reproducible sweeps and scripted visualization.",
            s["body"],
        )
    )

    # V
    story.append(P("V. EXPERIMENTAL SETUP", s["h1"]))
    story.append(P("A. Hardware and Reproducibility", s["h2"]))
    story.append(
        P(
            "Runs use CPU by default with fixed random seeds for NumPy and PyTorch. This favors "
            "reproducibility over wall-clock speed and avoids GPU nondeterminism as a confounder when "
            "comparing many nearby configurations.",
            s["body"],
        )
    )
    story.append(P("B. Grid Design", s["h2"]))
    story.append(
        P(
            "The principal sweep comprises 64 configurations exploring N ∈ {1,2,3,5}; global rounds "
            "(notably 5 and 10); local epochs E ∈ {1,2}; poison fraction f ∈ {0, 0.25, 0.40}; and poison "
            "mode ∈ {target, random_wrong} when f &gt; 0. Primary metrics are test accuracy, precision, "
            "recall, and F1. Selected representative scenarios additionally store full confusion matrices "
            "and TP/FP/FN/TN vectors.",
            s["body"],
        )
    )
    story.append(P("C. Descriptive Statistics of the Sweep", s["h2"]))
    story.append(
        P(
            "Across all 64 runs, accuracy has mean ≈ 0.907, standard deviation ≈ 0.014, minimum ≈ 0.856, "
            "and maximum ≈ 0.924. The relatively small standard deviation indicates that most configurations "
            "remain in a high-accuracy band; the lower tail is concentrated in heavily poisoned single-client "
            "settings. Mean F1 is ≈ 0.901 with a slightly larger spread, consistent with minority-class sensitivity.",
            s["body"],
        )
    )
    story.append(P("D. Evaluation Protocol and Artifacts", s["h2"]))
    story.append(
        P(
            "The test set is never poisoned and never used for training. Results are exported to CSV; static "
            "PNG figures and dashboards accompany the repository. Live dashboard: "
            "<link href='https://fl-research.vercel.app'>https://fl-research.vercel.app</link>.",
            s["body"],
        )
    )

    # VI Results
    story.append(P("VI. RESULTS", s["h1"]))
    story.append(P("A. Clean Training Behavior", s["h2"]))
    story.append(
        P(
            "Under clean labels, increasing R and E generally improves accuracy. Single-client clean "
            "training can slightly outperform multi-client FedAvg at the same budget because there is no "
            "averaging-induced damping; however, multi-client accuracy remains strong (typically &gt; 0.90). "
            "Figures 5 and 6 illustrate round and local-epoch effects on clean runs.",
            s["body"],
        )
    )
    fig(story, s, "03_rounds_vs_accuracy_clean.png", "Fig. 5. Effect of global rounds on clean-run accuracy.")
    fig(story, s, "06_local_epochs_clean_facets.png", "Fig. 6. Local-epoch effects for clean federated runs.")

    story.append(P("B. Clients and Poison Scenarios", s["h2"]))
    story.append(
        P(
            "Figure 7 compares best accuracy by client count and scenario. Figure 8 shows a "
            "clients-versus-poison heatmap of maximum accuracy. Figure 9 contrasts clean versus 40% "
            "target poisoning. Visually, the poison penalty is concentrated at N = 1 and nearly disappears at N = 5.",
            s["body"],
        )
    )
    fig(story, s, "01_accuracy_by_clients_scenario.png", "Fig. 7. Best accuracy by client count and poisoning scenario.")
    fig(story, s, "02_heatmap_clients_poison_pct.png", "Fig. 8. Max accuracy heatmap over clients versus poison percentage.")
    fig(story, s, "04_clean_vs_40pct_target.png", "Fig. 9. Clean versus 40% target poisoning (best accuracy per client count).")

    story.append(P("C. Best Configurations", s["h2"]))
    story.append(
        P(
            "Table I lists best observed metrics by condition. The most damaging cell is N = 1 with 40% "
            "target poison (accuracy ≈ 0.867 in the best-of-condition table; some individual runs fall near "
            "0.856). By contrast, N = 5 with 40% target poison remains essentially tied with the clean "
            "five-client best (≈ 0.914).",
            s["body"],
        )
    )
    story.append(P("TABLE I. Best test accuracy by client count and poison setting", s["tablecap"]))
    story.append(
        table(
            [
                ["Clients", "Poison", "Mode", "Acc.", "F1"],
                ["1", "0%", "—", "0.9237", "0.9212"],
                ["1", "25%", "target", "0.9022", "0.8881"],
                ["1", "40%", "target", "0.8666", "0.8329"],
                ["1", "40%", "random", "0.9180", "0.9127"],
                ["2", "0%", "—", "0.9214", "0.9177"],
                ["2", "40%", "target", "0.9188", "0.9132"],
                ["3", "0%", "—", "0.9181", "0.9152"],
                ["3", "40%", "target", "0.9178", "0.9135"],
                ["5", "0%", "—", "0.9139", "0.9119"],
                ["5", "40%", "target", "0.9139", "0.9117"],
            ],
            col_widths=[0.9 * inch, 0.9 * inch, 1.0 * inch, 0.9 * inch, 0.9 * inch],
        )
    )
    story.append(Spacer(1, 8))

    story.append(P("D. Mean Accuracy by Clients and Poison Fraction", s["h2"]))
    story.append(
        P(
            "Table II reports mean accuracy aggregated over rounds/epochs/modes in the sweep. The drop "
            "from f = 0 to f = 0.4 is large for N = 1 (≈ 0.916 → 0.887 mean) and small for N = 5 "
            "(≈ 0.908 → 0.905). Intermediate N values interpolate smoothly, supporting the 1/N dilution "
            "intuition of Section III-I.",
            s["body"],
        )
    )
    story.append(P("TABLE II. Mean test accuracy by clients and poison fraction", s["tablecap"]))
    story.append(
        table(
            [
                ["Clients", "f = 0", "f = 0.25", "f = 0.40"],
                ["1", "0.9158", "0.8981", "0.8873"],
                ["2", "0.9152", "0.9141", "0.9106"],
                ["3", "0.9133", "0.9127", "0.9082"],
                ["5", "0.9077", "0.9076", "0.9051"],
            ],
            col_widths=[1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch],
        )
    )
    story.append(Spacer(1, 8))

    story.append(P("E. Target versus Random-Wrong Poisoning", s["h2"]))
    story.append(
        P(
            "Target poisoning toward class 0 is typically more harmful than random-wrong poisoning at the "
            "same fraction in low-N regimes. Random-wrong noise is less coherently biased; target poisoning "
            "consistently teaches a majority-class prior. This distinction appears both in aggregate metrics "
            "and in confusion structure.",
            s["body"],
        )
    )

    story.append(P("F. Confusion Matrices: Clean Multi-Client", s["h2"]))
    story.append(
        P(
            "Figures 10–11 show five-client clean versus poisoned matrices. They are nearly alike: FedAvg "
            "dilution keeps decision boundaries stable. Class 0 dominates correct predictions; rare classes "
            "remain difficult; class 5 absorbs many off-diagonal errors even in clean training, indicating a "
            "structural confusion pattern of the representation/model rather than an artifact of poisoning alone.",
            s["body"],
        )
    )
    fig(story, s, "cm_5_clients__clean.png", "Fig. 10. Confusion matrix: 5 clients, clean training.")
    fig(story, s, "cm_5_clients__40pct_target_poison.png", "Fig. 11. Confusion matrix: 5 clients, 40% target poison on client 0.")

    story.append(P("G. Confusion Matrices: Single-Client Fragility", s["h2"]))
    story.append(
        P(
            "Figures 12–13 show the single-client contrast. After 40% target poisoning, substantial mass "
            "moves into predicted class 0 from true classes 4, 5, 6, and 7. Class 0 recall can remain high "
            "or even increase, while other classes suffer. Figures 14–15 make the same point via TP/FP/FN/TN "
            "bars: class 0 FP rises; FN rises for several non-zero classes. This is the empirical signature "
            "of majority-class targeted label noise under undiluted training.",
            s["body"],
        )
    )
    fig(story, s, "cm_1_client__clean.png", "Fig. 12. Confusion matrix: 1 client, clean training.")
    fig(story, s, "cm_1_client__40pct_target_poison.png", "Fig. 13. Confusion matrix: 1 client, 40% target poison.")
    fig(story, s, "tpfn_1_client__clean.png", "Fig. 14. Per-class TP/FP/FN/TN: 1 client, clean.")
    fig(story, s, "tpfn_1_client__40pct_target_poison.png", "Fig. 15. Per-class TP/FP/FN/TN: 1 client, 40% target poison.")

    story.append(P("H. Security Interpretation of Errors", s["h2"]))
    story.append(
        P(
            "From a SOC analyst’s perspective, elevated class 0 false positives after target poisoning mean "
            "attack flows are more often declared “majority/benign-like” (depending on label semantics), i.e., "
            "missed detections. Overall accuracy may still look acceptable because class 0 is frequent. This "
            "is why confusion-level reporting is mandatory for security ML evaluations and why Table I’s "
            "accuracy gaps understate operational harm at N = 1.",
            s["body"],
        )
    )

    # VII
    story.append(P("VII. DISCUSSION", s["h1"]))
    story.append(P("A. Implications for Deploying FL in IDS", s["h2"]))
    story.append(
        numbered(
            [
                "Prefer multiple independent data silos when possible; averaging provides baseline robustness to one bad silo under label noise.",
                "Prefer adequate R and E; under-trained models are both less accurate and potentially less stable under noise.",
                "Monitor per-class recall, not only accuracy, especially for rare attack types.",
                "Treat single-client “federated” training as essentially centralized: it inherits no dilution benefit.",
                "Use dashboards and confusion audits in experiment tracking so regressions in minority classes are visible.",
            ],
            s,
        )
    )
    story.append(P("B. Relationship to Robust FL Theory", s["h2"]))
    story.append(
        P(
            "Byzantine-robust aggregators were designed for stronger adversaries than our label-flip setting. "
            "Our results suggest that for a <i>single</i> weakly malicious client under IID equal shards, "
            "classical FedAvg already limits damage. That should not be read as a substitute for robust "
            "aggregation when clients can send arbitrary updates, when many clients are malicious, or when "
            "data are highly non-IID. Instead, dilution is a baseline phenomenon on top of which stronger "
            "defenses can be layered.",
            s["body"],
        )
    )
    story.append(P("C. What This Study Does Not Claim", s["h2"]))
    story.append(
        P(
            "We do not claim FedAvg is secure against adaptive model-poisoning or colluding clients. We do "
            "not claim MLP is the optimal IDS architecture. We do not claim IID shards represent real "
            "institutional heterogeneity. We do not claim that accuracy near 0.91 is sufficient for "
            "production detection without thresholding, ensembling, or human triage.",
            s["body"],
        )
    )
    story.append(P("D. Limitations", s["h2"]))
    story.append(
        bullets(
            [
                "<b>IID simulation:</b> random shards underestimate heterogeneity-induced drift.",
                "<b>Global scaler:</b> pooled normalization is a centralized shortcut.",
                "<b>Single architecture:</b> no comparison to trees, CNNs on raw packets, or sequence models.",
                "<b>Undefended aggregation:</b> no median/Krum/DP baselines in the sweep.",
                "<b>Label semantics:</b> class indices are used without exhaustive attack taxonomy narration.",
                "<b>No systems noise:</b> no stragglers, partial participation, or compression artifacts.",
            ],
            s,
        )
    )
    story.append(P("E. Threats to Validity", s["h2"]))
    story.append(
        P(
            "Random seeds, hyperparameter choices (Adam learning rate, batch size), and the particular "
            "augmented dataset release could shift absolute numbers. Relative conclusions—dilution with N, "
            "severity of target poison at N = 1, need for confusion analysis—were consistent across the "
            "sweep’s internal variation, which increases confidence in the qualitative story.",
            s["body"],
        )
    )
    story.append(P("F. Future Work", s["h2"]))
    story.append(
        numbered(
            [
                "Non-IID label and feature skew across clients.",
                "Robust aggregators and anomaly detection on updates.",
                "Differentially private FL for formal privacy guarantees.",
                "Cost-sensitive losses or resampling for rare classes.",
                "Multi-poisoner and model-replacement attacks.",
                "Deployment in a real FL framework with networked workers.",
                "Richer backbone models and calibrated probability outputs for SOC workflows.",
            ],
            s,
        )
    )

    # VIII
    story.append(P("VIII. CONCLUSION", s["h1"]))
    story.append(
        P(
            "We presented an expanded theoretical and experimental study of FedAvg for multi-class traffic "
            "classification on CIC–UNSW-NB15. Starting from the global empirical risk and its client-wise "
            "decomposition, we formalized local optimization, size-weighted aggregation, communication "
            "scaling, class-imbalance metrics, and a single-client label-poisoning adversary. Empirically, "
            "among 64 configurations, clean training benefits from more rounds and local epochs; targeted "
            "poisoning is highly damaging when N = 1 and largely diluted when N ≥ 2; and confusion matrices "
            "reveal majority-class bias that accuracy can mask. These findings support a practical message: "
            "federated collaboration with multiple honest clients, sufficient training, and per-class "
            "monitoring forms a stronger baseline than accuracy-only evaluation of single-silo training under "
            "label noise.",
            s["body"],
        )
    )

    story.append(P("ACKNOWLEDGMENT", s["h1"]))
    story.append(
        P(
            "The author thanks the maintainers of CIC–UNSW-NB15 related public datasets and the developers "
            "of PyTorch, NumPy, pandas, and scikit-learn.",
            s["body"],
        )
    )

    story.append(P("REFERENCES", s["h1"]))
    refs = [
        "[1] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Proc. AISTATS, 2017, pp. 1273–1282.",
        "[2] P. Kairouz et al., “Advances and open problems in federated learning,” Found. Trends Mach. Learn., vol. 14, no. 1–2, pp. 1–210, 2021.",
        "[3] S. A. Rahman, H. Tout, C. Talhi, and A. Mourad, “Internet of Things intrusion detection: Centralized, on-device, or federated learning?” IEEE Network, vol. 34, no. 6, pp. 310–317, 2020.",
        "[4] E. Bagdasaryan, A. Veit, Y. Hua, D. Estrin, and V. Shmatikov, “How to backdoor federated learning,” in Proc. AISTATS, 2020, pp. 2938–2948.",
        "[5] V. Tolpegin, S. Truex, M. E. Gursoy, and L. Liu, “Data poisoning attacks against federated learning systems,” in Proc. ESORICS, 2020, pp. 480–501.",
        "[6] T. Li, A. K. Sahu, A. Talwalkar, and V. Smith, “Federated learning: Challenges, methods, and future directions,” IEEE Signal Process. Mag., vol. 37, no. 3, pp. 50–60, 2020.",
        "[7] K. Bonawitz et al., “Towards federated learning at scale: System design,” in Proc. MLSys, 2019.",
        "[8] N. Moustafa and J. Slay, “UNSW-NB15: A comprehensive data set for network intrusion detection systems,” in Proc. MilCIS, 2015, pp. 1–6.",
        "[9] P. Blanchard, E. M. El Mhamdi, R. Guerraoui, and J. Stainer, “Machine learning with adversaries: Byzantine tolerant gradient descent,” in Proc. NeurIPS, 2017.",
        "[10] D. Yin, Y. Chen, R. Kannan, and P. Bartlett, “Byzantine-robust distributed learning: Towards optimal statistical rates,” in Proc. ICML, 2018.",
    ]
    for r in refs:
        story.append(P(r, s["ref"]))

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Times-Roman", 8)
        canvas.drawCentredString(letter[0] / 2.0, 0.4 * inch, f"{doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print("Wrote", OUT, "size", OUT.stat().st_size)


if __name__ == "__main__":
    build()
