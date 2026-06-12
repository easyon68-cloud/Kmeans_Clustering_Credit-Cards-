# Kmeans_Clustering_Credit-Cards-
Customer Segmentation of Credit Card Users — K-Means Clustering

Unsupervised Machine Learning | Customer Analytics | K-Means | PCA Visualization




📌 Project Overview

This project applies K-Means Clustering to segment 8,950 credit card holders into distinct behavioral groups based on their spending patterns, balance habits, cash advance usage, payment behavior, and credit utilization.

The goal is to transform raw transaction data into actionable customer personas — enabling banks and financial institutions to deliver targeted marketing, improve risk assessment, and design personalized financial products for each segment.

The pipeline follows a complete, production-quality ML workflow:

Raw Data → EDA → Missing Value Treatment → Feature Scaling →
Elbow Method → Silhouette Validation → Final Model → PCA Visualization →
Cluster Profiling → Business Labeling → Export


📂 Dataset

PropertyDetailFileCC GENERAL.csvRows8,950 credit card holdersColumns18 featuresSourceKaggle — Credit Card Dataset for Clustering

Feature Dictionary

ColumnTypeDescriptionCUST_IDCategoricalUnique customer identifier (dropped before modeling)BALANCEFloatRevolving balance in the accountBALANCE_FREQUENCYFloat (0–1)How frequently balance is updatedPURCHASESFloatTotal purchase amountONEOFF_PURCHASESFloatMaximum single-transaction purchaseINSTALLMENTS_PURCHASESFloatTotal installment purchase amountCASH_ADVANCEFloatTotal cash advance drawnPURCHASES_FREQUENCYFloat (0–1)Frequency of purchasesONEOFF_PURCHASES_FREQUENCYFloat (0–1)Frequency of one-off purchasesPURCHASES_INSTALLMENTS_FREQUENCYFloat (0–1)Frequency of installment purchasesCASH_ADVANCE_FREQUENCYFloat (0–1)Frequency of cash advancesCASH_ADVANCE_TRXIntegerNumber of cash advance transactionsPURCHASES_TRXIntegerNumber of purchase transactionsCREDIT_LIMITFloatCredit card limit (1 missing value)PAYMENTSFloatTotal payment amount madeMINIMUM_PAYMENTSFloatMinimum payment made (313 missing values)PRC_FULL_PAYMENTFloat (0–1)Percentage of months with full paymentTENUREIntegerMonths of credit card service


🧹 Data Preprocessing

Missing Value Treatment

ColumnMissing CountStrategyCREDIT_LIMIT1Convert to numeric → fill with column medianMINIMUM_PAYMENTS313 (3.5%)Convert to numeric → fill with column median

Why median over mean?
Both columns exhibit strong right-skew with outliers. The median is resistant to extreme values and represents the "typical customer" more accurately than the inflated mean.

Feature Engineering


CUST_ID (string identifier) is dropped — carries no behavioral information and cannot be used in distance calculations
All 17 remaining columns converted to float64 before scaling


Standard Scaling

All 17 features are normalized using StandardScaler (formula: z = (x − μ) / σ):

ReasonDetailScale disparityBALANCE ranges 0–19,000; PURCHASES_FREQUENCY ranges 0–1Distance fairnessWithout scaling, high-magnitude features dominate Euclidean distancesCluster qualityScaling ensures K-Means centroids reflect all features equally

After scaling: mean ≈ 0, std ≈ 1 for every column. Shape confirmed: (8,950 × 17).


📐 Optimal K Selection

Method 1 — Elbow Method (Inertia / WCSS)

Inertia (Within-Cluster Sum of Squares) was computed for K = 2 to 10:

KInertiaNote2127,7853111,975499,062← Elbow point591,490Diminishing return684,827……

The curve bends sharply at K=4 — additional clusters provide progressively smaller improvements.

Method 2 — Silhouette Score

Silhouette scores were computed for K = 2 to 10:

KSilhouette ScoreNote20.215730.2403Mathematically highest40.2004Chosen ✅50.1975……

Final Decision: K = 4

K=4 was chosen over K=3 because:


It aligns with the elbow method's clear signal
It produces 4 distinct business personas vs. only 3 with K=3
The VIP High-Spenders and Cash-Advance Reliant segments are meaningfully different and require different business responses
The marginal silhouette difference (0.2403 vs. 0.2004) is small relative to the gain in business interpretability



🤖 Final Model Configuration

pythonKMeans(
    n_clusters=4,
    init='k-means++',   # Smart seeding — distributes initial centroids
    n_init=20,           # 20 independent runs → best result kept
    max_iter=300,        # Max convergence iterations per run
    random_state=42      # Reproducibility
)

ParameterValueRationaleinit='k-means++'Smart initAvoids poor local minima vs. random initializationn_init=2020 runsHigher than exploration (10) — ensures global optimummax_iter=300StandardSufficient for convergence on this datasetrandom_state=42FixedReproducible results for reporting

Final Metrics:

MetricValueInertia (WCSS)99,062Silhouette Score0.2004


📊 PCA Visualization

Since K-Means operates in 17 dimensions, Principal Component Analysis (PCA) compresses the data to 2 dimensions for visualization:


PCA(n_components=2) applied to X_scaled (17 features → 2 principal components)
Cluster labels from the 17-D model are transferred to the 2D plot
Cluster centroids are projected through the same PCA transform and overlaid as black ✕ markers
PC1 and PC2 together typically capture 30–50% of total variance for this dataset



📌 PCA is used only for visualization — clustering was performed on the full 17-dimensional space.




🎯 Cluster Profiles & Business Segments

Cluster Summary Table

ClusterBalancePurchasesCash AdvanceCredit LimitFull Payment %CustomersShare0$1,012$270$597$3,279Low3,97744.4%1$3,551$7,682$654$9,697Moderate4094.6%2$4,602$502$4,522$7,546Low1,19713.4%3$895$1,236$211$4,213High3,36737.6%

Segment Personas

💤 Cluster 0 — Low-Activity / Dormant (44.4%)


Lowest balance, purchases, and credit limit of all segments
Minimal engagement with card — likely using it only occasionally
Business Action: Re-engagement campaigns, introductory cashback offers, spending incentives


🌟 Cluster 1 — VIP High-Spenders (4.6%)


Highest purchases ($7,682 avg) and highest credit limit ($9,697)
High balance but also high payments — active, valuable customers
Business Action: Premium loyalty programs, exclusive rewards, travel perks, retention priority


💳 Cluster 2 — Cash-Advance Reliant (13.4%)


Extremely high cash advances ($4,522 avg) and high balance ($4,602)
Low purchases — using card primarily as a cash source, not for shopping
Business Action: Financial counseling programs, debt management offers, risk monitoring


✅ Cluster 3 — Responsible Regular Users (37.6%)


Moderate purchases ($1,236), low cash advance, good payment behavior
Stable, reliable segment with room for growth
Business Action: Credit limit increases, product upsells, installment plan promotions



📁 Project Structure

📦 Kmeans_Creditcard_Segmentation/
├── 📓 Kmeans_Creditcard_Assignment.ipynb          ← Original notebook
├── 📓 Kmeans_Creditcard_Annotated.ipynb           ← Fully annotated notebook
├── 📄 CC GENERAL.csv                              ← Raw dataset (8,950 × 18)
├── 📄 CC_Clustered_Results.csv                    ← Output with Cluster + Cluster_Name
└── 📄 README.md                                   ← This file


🔧 Libraries & Requirements

pythonpandas          >= 1.3     # Data loading, cleaning, groupby profiling
numpy           >= 1.21    # Array operations, argmax for best-k selection
matplotlib      >= 3.4     # All chart rendering
seaborn         >= 0.11    # Correlation heatmap, boxplots
scikit-learn    >= 0.24    # KMeans, StandardScaler, PCA, silhouette_score
scipy           >= 1.7     # Hierarchical clustering utilities (linkage, dendrogram)

Install All Dependencies

bashpip install pandas numpy matplotlib seaborn scikit-learn scipy


🚀 How to Run


Clone or download the repository
Place CC GENERAL.csv in the working directory (or update the path in the Load Dataset cell)
Open the notebook in Google Colab or Jupyter Notebook
Run all cells top to bottom — each step depends on the previous



⚠️ Update the dataset path in the Load cell for local execution:

pythondf = pd.read_csv('CC GENERAL.csv')



Expected Runtime

StepApprox. TimeEDA + Preprocessing< 10 secondsElbow Method (K=2–10)30–60 secondsSilhouette Loop (K=2–10)60–120 secondsFinal Model Training< 10 secondsPCA + Visualization< 5 seconds


📈 Key Findings


K=4 clusters were identified as optimal through combined elbow method and silhouette analysis
44.4% of customers are dormant — the largest segment and a primary target for re-engagement
4.6% are VIP High-Spenders — a small but disproportionately valuable group
13.4% are Cash-Advance Reliant — a financially vulnerable segment requiring careful risk management
PURCHASES and CASH_ADVANCE are the two most discriminating features across clusters
k-means++ initialization with n_init=20 ensures robust, stable cluster assignments



💼 Business Applications

Use CaseHow Clustering HelpsTargeted MarketingDifferent ad campaigns for each of the 4 behavioral segmentsRisk AssessmentCluster 2 (Cash-Advance Reliant) flags high-risk accounts for early interventionProduct DesignCluster 1 (VIP) → premium cards; Cluster 3 (Regular) → cashback cardsChurn PreventionCluster 0 (Dormant) → proactive outreach before accounts closeCredit DecisionsCluster 3's responsible behavior supports credit limit increase approvals


👤 Author

K-Means Credit Card Customer Segmentation Assignment
Machine Learning | Unsupervised Learning | Financial Analytics
