"""
ESTILO COMPARTILHADO DAS CENAS
==============================

Tudo que é comum às cenas fica aqui: cores, fonte, funções para criar texto,
fórmulas, "pixels" e números no formato brasileiro.

Mude uma cor aqui e ela muda em TODAS as cenas.
"""
import json
from pathlib import Path

import numpy as np
from manim import (BOLD, DOWN, LEFT, ORIGIN, RIGHT, UP, FadeOut, ManimColor, MathTex,
                   Rectangle, Square, Text, VGroup, interpolate_color)
from manim_slides import Slide

# =====================================================================
# CORES
# =====================================================================
COR_FUNDO = "#0F1117"          # fundo da tela
COR_TEXTO = "#E8E8E8"          # texto principal
COR_SUAVE = "#8A8F98"          # texto secundário / legendas / bordas
COR_BENIGNO = "#4C9BE8"        # azul = benigno
COR_MALIGNO = "#F2994A"        # laranja = maligno
COR_DESTAQUE = "#F2C94C"       # amarelo para chamar atenção
COR_RUIM = "#EB5757"           # vermelho para "isso é um problema"
COR_BOM = "#6FCF97"            # verde para "isso funciona"
# Barras do ranking: do amarelo (mais importante) ao roxo (menos importante)
GRADIENTE_RANKING = ["#F2C94C", "#F2994A", "#EB5757", "#9B51E0", "#5B3E96"]

# Escala de cinza dos "pixels": z = -LIMITE_Z vira preto, z = +LIMITE_Z vira branco
LIMITE_Z = 2.0
PIXEL_ESCURO = "#141414"
PIXEL_CLARO = "#FFFFFF"
COR_BORDA_PIXEL = "#3A3F4A"    # contorno de cada pixel

FONTE = "Helvetica Neue"       # troque por outra fonte instalada no seu Mac, se quiser

# =====================================================================
# DADOS REAIS (gerados por gerar_dados.py)
# =====================================================================
DADOS = json.loads((Path(__file__).parent / "dados" / "mol_dados.json").read_text())
N_FEATURES = len(DADOS["feature_names"])


# =====================================================================
# FUNÇÕES AUXILIARES
# =====================================================================
def texto(conteudo, tamanho=28, cor=COR_TEXTO, **kwargs):
    """Texto com a fonte e a cor padrão. tamanho = tamanho da fonte."""
    return Text(conteudo, font=FONTE, font_size=tamanho, color=cor, **kwargs)


def formula(*partes, tamanho=40, cor=COR_TEXTO):
    """
    Fórmula em LaTeX. Pode receber várias partes, por exemplo:
        formula(r"z_{ij}", "=", r"\\frac{x_{ij}-\\mu_j}{\\sigma_j}")
    e depois colorir uma parte: f[2].set_color(COR_DESTAQUE)
    """
    return MathTex(*partes, font_size=tamanho, color=cor)


def br(valor, casas=2):
    """Número no formato brasileiro (vírgula decimal). br(0.786, 3) -> '0,786'."""
    return f"{valor:.{casas}f}".replace(".", ",")


def tex(valor, casas=2):
    """Número para dentro de fórmulas LaTeX (a vírgula precisa de {,})."""
    return f"{valor:.{casas}f}".replace(".", "{,}")


def valor_bruto(v):
    """Formata um valor bruto da tabela com ~4 algarismos significativos."""
    if abs(v) >= 100:
        s = f"{v:.1f}"
    elif abs(v) >= 10:
        s = f"{v:.2f}"
    elif abs(v) >= 1:
        s = f"{v:.3f}"
    else:
        s = f"{v:.4f}"
    return s.replace(".", ",")


def nome_curto(nome):
    """'concave points_worst' -> 'concave points worst'."""
    return nome.replace("_", " ")


def cor_do_pixel(z):
    """Converte um z-score num tom de cinza (baixo = escuro, alto = claro)."""
    t = (np.clip(z, -LIMITE_Z, LIMITE_Z) + LIMITE_Z) / (2 * LIMITE_Z)
    return interpolate_color(ManimColor(PIXEL_ESCURO), ManimColor(PIXEL_CLARO), t)


def pixel(z, lado):
    """Um quadradinho cinza representando um valor padronizado."""
    return (Square(side_length=lado)
            .set_fill(cor_do_pixel(z), opacity=1)
            .set_stroke(COR_BORDA_PIXEL, width=1.5))  # borda cinza: pixels pretos não somem no fundo


def imagem_mol(z, lado=0.3, linhas=5, colunas=6, ordem=None):
    """
    Monta a imagem de um paciente: as features entram na ordem `ordem`
    (padrão: ranking do Map euclidiano) e preenchem a grade linha a linha.
    """
    ordem = DADOS["ordem"] if ordem is None else ordem
    grade = VGroup(*[pixel(z[ordem[k]], lado) for k in range(linhas * colunas)])
    return grade.arrange_in_grid(rows=linhas, cols=colunas, buff=0)


def moldura(mob, cor, buff=0.06):
    """Retângulo colorido em volta de um objeto (ex.: imagem benigna em azul)."""
    return Rectangle(width=mob.width + 2 * buff, height=mob.height + 2 * buff,
                     stroke_color=cor, stroke_width=3).move_to(mob)


class CenaBase(Slide):
    """Classe base: fundo escuro + atalho para limpar a tela."""

    def setup(self):
        super().setup()
        self.camera.background_color = ManimColor(COR_FUNDO)

    def limpar(self, *manter):
        """Faz FadeOut de tudo que está na tela, exceto os objetos em `manter`."""
        sair = [m for m in self.mobjects if m not in manter]
        if sair:
            self.play(*[FadeOut(m) for m in sair])

    def cabecalho(self, conteudo, tamanho=34, cor=COR_TEXTO):
        """Título no topo da tela."""
        return texto(conteudo, tamanho, cor).to_edge(UP, buff=0.35)
