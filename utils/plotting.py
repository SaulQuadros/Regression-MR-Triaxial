import numpy as np
import plotly.graph_objs as go

def plot_3d_surface(df, model, energy_col="MR"):
    # malhas 1D para garantir semântica clara (Plotly: z.shape == (len(y), len(x)))
    s3 = np.linspace(df["σ3"].min(), df["σ3"].max(), 30)
    sd = np.linspace(df["σd"].min(), df["σd"].max(), 30)
    s3g, sdg = np.meshgrid(s3, sd, indexing="xy")  # shapes: (len(sd), len(s3))
    Xg = np.c_[s3g.ravel(order="C"), sdg.ravel(order="C")]
    
    MRg = model.predict(Xg)
    MRg = MRg.reshape(s3g.shape, order="C")  # => (len(sd), len(s3))

    # Use x=s3 (1D), y=sd (1D), z=MRg (2D) para evitar ambiguidades
    fig = go.Figure(data=[go.Surface(x=s3, y=sd, z=MRg, colorscale='Viridis')])

    # Pontos observados
    fig.add_trace(go.Scatter3d(
        x=df["σ3"], y=df["σd"], z=df[energy_col],
        mode='markers', marker=dict(size=5, color='red'), name="Dados"
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='σ₃ (MPa)',
            yaxis_title='σ_d (MPa)',
            zaxis_title='MR (MPa)'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig
