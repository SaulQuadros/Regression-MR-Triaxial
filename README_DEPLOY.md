
# Branch: kaleido-export (NO APT)

Este branch remove `packages.txt` e usa pins conservadores para evitar erros de instalação no Streamlit Cloud.
O Kaleido deve funcionar com o Chromium embutido no próprio pacote (sem APT).

## Passos
1) Criar o branch `kaleido-export` no GitHub (ou `kaleido-export-noapt` se preferir).
2) Subir `app.py`, `requirements.txt`, `runtime.txt`.
3) Redeploy no Streamlit apontando para esse branch.

Se ainda falhar, compartilhe o log do pip (Manage app → Logs) — geralmente é conflito do `numpy 2.x`.
