### LangChain: 

1. https://www.youtube.com/watch?v=AOQyRiwydyo&list=PL7SI75SatyIKbXB8_GOpstB_PznQeMVPV&t=2416s

### LangGraph: 

1. https://www.youtube.com/watch?v=dIb-DujRNEo&list=PL7SI75SatyIKbXB8_GOpstB_PznQeMVPV&index=2
2. https://www.youtube.com/watch?v=DtW_Lc9hYoU&list=PL7SI75SatyIKbXB8_GOpstB_PznQeMVPV&index=3

---

### Steps for Setup:

```python
sudo apt update
sudo apt install pipx -y
pipx ensurepath

pipx install uv

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

uv --version

uv sync
```