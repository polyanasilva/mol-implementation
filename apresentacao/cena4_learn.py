"""
CENA 4 — Etapa LEARN: a CNN, camada por camada
===============================================

Roteiro (cada método slide_... termina num `self.next_slide()`, onde a apresentação pausa):
   1. Título
   2. O que entra: uma imagem por paciente (e a observação sobre a imagem 16x1 do ABC)
   3. A arquitetura completa, com o formato dos dados em cada camada
   4. Convolução: a fórmula, uma conta com números reais e o filtro deslizando
   5. ReLU e os 32 filtros
   6. Max-pooling
   7. Camadas densas e a sigmoide (de 288 números a uma probabilidade)
   8. Função de perda: entropia cruzada
   9. Adam e a curva de treino real
  10. Previsões no conjunto de teste e a matriz de confusão
  11. Saída da etapa

COMO RODAR (de dentro da pasta apresentacao/):
    .venv/bin/manim-slides render -ql cena4_learn.py Cena4Learn   # prévia rápida
    .venv/bin/manim-slides render cena4_learn.py Cena4Learn       # 1080p
    .venv/bin/manim-slides present Cena4Learn                     # apresentar

Todos os números vêm de uma CNN REAL (dados/mol_dados.json, chave "learn"): a rede do artigo
treinada nas imagens 5x6 (30 features na ordem do Map), 100 épocas, taxa de aprendizado 0,001.
As contas mostradas (convolução, ReLU, pooling, saída) foram refeitas à mão e conferidas com o Keras.
"""
import numpy as np
from manim import *

from estilo import (COR_BENIGNO, COR_BOM, COR_DESTAQUE, COR_FUNDO, COR_MALIGNO, COR_RUIM,
                    COR_SUAVE, COR_TEXTO, DADOS, CenaBase, br, cor_do_pixel, formula, pixel, tex, texto)

L = DADOS["learn"]
OPT = DADOS["optimize"]
H, W = L["forma"]
IMG = np.array(L["paciente"]["imagem"])          # imagem 5x6 do paciente de exemplo (maligno)
FILTRO = L["filtro"]
PESOS = np.array(FILTRO["pesos"])
PRE = np.array(FILTRO["pre"])                    # saída da convolução (antes do ReLU)
RELU = np.array(FILTRO["relu"])
POOL = np.array(FILTRO["pool"])

LADO = 0.6                                       # tamanho das células com número


def celula(valor, cor_fundo, lado=LADO, casas=2, tamanho=15, cor_texto=None):
    """Quadrado colorido com um número dentro."""
    q = Square(side_length=lado).set_fill(cor_fundo, 1).set_stroke("#3A3F4A", 1.5)
    if cor_texto is None:
        r_, g_, b_ = ManimColor(cor_fundo).to_rgb()
        cor_texto = "#111111" if 0.299 * r_ + 0.587 * g_ + 0.114 * b_ > 0.55 else COR_TEXTO
    t = texto(br(valor, casas), tamanho, cor_texto)
    if t.width > lado - 0.08:
        t.scale_to_fit_width(lado - 0.08)
    return VGroup(q, t.move_to(q))


def cor_ativacao(v, vmax):
    """Mapa de ativação: positivo = amarelo (mais forte = mais claro), negativo = vermelho."""
    if v < 0:
        return interpolate_color(ManimColor(COR_FUNDO), ManimColor(COR_RUIM), min(-v / vmax * 2 + 0.3, 1))
    return interpolate_color(ManimColor(COR_FUNDO), ManimColor(COR_DESTAQUE), min(v / vmax, 1))


def cor_peso(v, vmax):
    return interpolate_color(ManimColor(COR_FUNDO), ManimColor(COR_BENIGNO), min(abs(v) / vmax, 1))


def grade_numeros(matriz, cor_fn, lado=LADO, casas=2, tamanho=15):
    """Grade de células numeradas a partir de uma matriz 2D."""
    m = np.array(matriz)
    g = VGroup(*[celula(m[r, c], cor_fn(m[r, c]), lado, casas, tamanho)
                 for r in range(m.shape[0]) for c in range(m.shape[1])])
    return g.arrange_in_grid(rows=m.shape[0], cols=m.shape[1], buff=0)


class Cena4Learn(CenaBase):
    def construct(self):
        self.slide_1_titulo()
        self.slide_2_entrada()
        self.slide_3_arquitetura()
        self.slide_4_convolucao()
        self.slide_5_relu_e_filtros()
        self.slide_6_max_pooling()
        self.slide_7_densas_e_sigmoide()
        self.slide_8_perda()
        self.slide_9_adam_e_treino()
        self.slide_10_teste()
        self.slide_11_saida()

    # =================================================================
    def trocar_cabecalho(self, conteudo, tamanho=32, cor=COR_TEXTO):
        novo = self.cabecalho(conteudo, tamanho, cor)
        if getattr(self, "titulo", None) is not None and self.titulo in self.mobjects:
            self.play(ReplacementTransform(self.titulo, novo))
        else:
            self.play(FadeIn(novo))
        self.titulo = novo

    def limpar_tudo(self):
        self.limpar()
        self.titulo = None

    # =================================================================
    def slide_1_titulo(self):
        etapa = texto("ETAPA 3", 30, COR_DESTAQUE)
        titulo = texto("LEARN", 96, weight=BOLD)
        sub = texto("A CNN aprende a classificar as imagens", 32, COR_SUAVE)
        VGroup(etapa, titulo, sub).arrange(DOWN, buff=0.35)
        self.play(FadeIn(etapa, shift=DOWN * 0.3))
        self.play(Write(titulo))
        self.play(FadeIn(sub, shift=UP * 0.3))
        self.next_slide()
        self.limpar()

    # =================================================================
    def slide_2_entrada(self):
        self.trocar_cabecalho("O que entra: uma imagem por paciente")

        def imagem(z, cor, nome):
            g = VGroup(*[pixel(v, 0.5) for v in np.array(z).ravel()]).arrange_in_grid(rows=H, cols=W, buff=0)
            g.add(SurroundingRectangle(g, color=cor, buff=0.04, stroke_width=4))
            return VGroup(g, texto(nome, 24, cor)).arrange(DOWN, buff=0.2)

        ben = imagem(L["benigno"]["imagem"], COR_BENIGNO, "benigno  (y = 0)")
        mal = imagem(L["paciente"]["imagem"], COR_MALIGNO, "maligno  (y = 1)")
        par = VGroup(ben, mal).arrange(RIGHT, buff=1.2).move_to(UP * 0.5)
        tensor = formula(r"\text{entrada: } N \times h \times w \times 1 \;=\; 455 \times 5 \times 6 \times 1",
                         tamanho=36).next_to(par, DOWN, buff=0.4)
        nota_t = texto("455 pacientes de treino · imagem 5×6 · 1 canal (tons de cinza)", 20, COR_SUAVE) \
            .next_to(tensor, DOWN, buff=0.15)
        self.play(FadeIn(ben, shift=RIGHT * 0.3), FadeIn(mal, shift=LEFT * 0.3))
        self.play(Write(tensor), FadeIn(nota_t))
        self.next_slide()

        # Observação: a solução do ABC é 16x1
        z_m = OPT["pacientes_ordenados"]["maligno"]
        mascara = OPT["otimizadores"]["abc"]["mascara"]
        tira = VGroup(*[pixel(z_m[k], 0.2) for k in range(30) if mascara[k]]).arrange(DOWN, buff=0)
        obs = VGroup(
            texto("Observação: a imagem escolhida pelo ABC é 16×1.", 20, COR_DESTAQUE),
            texto("Numa imagem de 1 coluna, o filtro 3×3 só enxerga vizinhos na vertical:", 18, COR_SUAVE),
            texto("vira uma convolução de 1 dimensão.", 18, COR_SUAVE),
            texto("Para mostrar a convolução em 2D, esta cena usa a imagem 5×6", 18, COR_TEXTO),
            texto("com as 30 features (a melhor CNN nos testes: 0,974).", 18, COR_TEXTO),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        bloco = VGroup(tira, obs).arrange(RIGHT, buff=0.5)
        self.play(FadeOut(VGroup(tensor, nota_t)), par.animate.scale(0.7).move_to(LEFT * 4.3 + DOWN * 0.2))
        bloco.next_to(par, RIGHT, buff=0.45)
        if bloco.get_right()[0] > 6.8:
            bloco.scale_to_fit_width(6.8 - bloco.get_left()[0], about_edge=LEFT)
        self.play(FadeIn(bloco))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_3_arquitetura(self):
        self.trocar_cabecalho("A arquitetura do artigo: pequena, com 1 convolução")
        p = L["parametros"]
        n_flat = 3 * 3 * 32
        etapas = [
            ("Entrada", "imagem", "5 × 6 × 1", "", COR_TEXTO),
            ("Convolução", "32 filtros 3×3\n+ ReLU", "5 × 6 × 32", f"{p['conv']} pesos", COR_DESTAQUE),
            ("Max-pooling", "janelas 2×2", "3 × 3 × 32", "0 pesos", COR_DESTAQUE),
            ("Flatten", "enfileira tudo", f"{n_flat}", "0 pesos", COR_SUAVE),
            ("Densa", f"{L['n_oculta']} neurônios\n+ ReLU", f"{L['n_oculta']}", f"{p['densa']:,} pesos".replace(",", "."), COR_BENIGNO),
            ("Saída", "1 neurônio\n+ sigmoide", "1 (probabilidade)", f"{p['saida']} pesos", COR_MALIGNO),
        ]
        caixas = VGroup()
        for nome, desc, forma, pesos, cor in etapas:
            r = RoundedRectangle(width=1.9, height=1.7, corner_radius=0.12, stroke_color=cor, stroke_width=3)
            conteudo = VGroup(texto(nome, 20, cor, weight=BOLD), texto(desc, 15, COR_TEXTO, line_spacing=1.0)) \
                .arrange(DOWN, buff=0.15).move_to(r)
            saida = texto(forma, 17, COR_TEXTO).next_to(r, DOWN, buff=0.2)
            n_p = texto(pesos, 15, COR_SUAVE).next_to(saida, DOWN, buff=0.12)
            caixas.add(VGroup(r, conteudo, saida, n_p))
        caixas.arrange(RIGHT, buff=0.38, aligned_edge=UP).move_to(UP * 0.4)
        setas = VGroup(*[Arrow(caixas[i][0].get_right(), caixas[i + 1][0].get_left(), buff=0.04, color=COR_SUAVE,
                               stroke_width=3, max_tip_length_to_length_ratio=0.4) for i in range(len(caixas) - 1)])
        rot_forma = texto("formato da saída de cada camada →", 15, COR_SUAVE).next_to(caixas[0][2], LEFT, buff=0.1)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in caixas], lag_ratio=0.25), run_time=3)
        self.play(FadeIn(setas))
        total = p["conv"] + p["densa"] + p["saida"]
        resumo = VGroup(
            texto(f"Total: {total:,} pesos aprendidos".replace(",", "."), 26, COR_DESTAQUE),
            texto("Camada densa do artigo: (entradas + saídas) / 2 = (30 + 1) / 2 → 16 neurônios", 19, COR_SUAVE),
            texto("Com a imagem 16×1 do ABC: 16×1×32 → 8×1×32 → 256 → 9 → 1  (2.643 pesos)", 19, COR_SUAVE),
        ).arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(resumo))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_4_convolucao(self):
        self.trocar_cabecalho("Convolução: um filtro 3×3 desliza pela imagem")
        f = formula(r"a_k[r, c] \;=\; b_k + \sum_{u=-1}^{1}\sum_{v=-1}^{1} W_k[u, v]\; I[r+u,\, c+v]",
                    tamanho=34).next_to(self.titulo, DOWN, buff=0.2)
        self.play(Write(f))

        # Imagem do paciente com os valores (z-scores) e a borda de zeros do padding "same"
        img = grade_numeros(IMG, cor_do_pixel).move_to(LEFT * 3.9 + DOWN * 0.35)
        borda = VGroup(*[Square(LADO).set_stroke(COR_SUAVE, 1, opacity=0.5) for _ in range((H + 2) * (W + 2))]) \
            .arrange_in_grid(rows=H + 2, cols=W + 2, buff=0).move_to(img)
        zeros = VGroup(*[texto("0", 13, COR_SUAVE).move_to(borda[i]) for i in range(len(borda))
                         if i // (W + 2) in (0, H + 1) or i % (W + 2) in (0, W + 1)])
        rot_img = texto("imagem do paciente (valores z)", 16, COR_SUAVE).next_to(img, DOWN, buff=0.12)
        self.play(FadeIn(img), FadeIn(rot_img))
        self.next_slide()

        # O filtro, com os pesos REAIS aprendidos
        vmax_w = np.abs(PESOS).max()
        filtro = grade_numeros(PESOS, lambda v: cor_peso(v, vmax_w)).move_to(RIGHT * 0.55 + UP * 0.25)
        rot_f = VGroup(texto(f"filtro W (o nº {FILTRO['indice'] + 1} de 32)", 16, COR_BENIGNO),
                       texto(f"pesos aprendidos · viés b = {br(FILTRO['bias'], 3)}", 14, COR_SUAVE)) \
            .arrange(DOWN, buff=0.06).next_to(filtro, DOWN, buff=0.12)
        self.play(FadeIn(filtro), FadeIn(rot_f))

        # Janela 3x3 na posição do exemplo
        r0, c0 = FILTRO["posicao_exemplo"]

        def janela(r, c):
            return Square(3 * LADO, stroke_color=COR_DESTAQUE, stroke_width=5).move_to(img[r * W + c])

        jan = janela(r0, c0)
        self.play(Create(jan))
        x = np.pad(IMG, 1)[r0:r0 + 3, c0:c0 + 3]
        termos = " + ".join(rf"{tex(PESOS[u, v], 2)} \cdot {tex(x[u, v], 2)}" for u, v in [(0, 0), (0, 1), (0, 2)])
        conta = formula(rf"a[{r0 + 1},{c0 + 1}] = {tex(FILTRO['bias'], 3)} + \left({termos} + \dots\right) =",
                        tex(PRE[r0, c0], 3), tamanho=27).to_edge(DOWN, buff=0.3)
        conta[1].set_color(COR_DESTAQUE)
        expl = texto("cada pixel da janela × o peso na mesma posição, soma tudo (9 produtos) e junta o viés",
                     17, COR_SUAVE).next_to(conta, UP, buff=0.12)
        self.play(Write(conta), FadeIn(expl))

        # Mapa de saída: começa vazio, a célula do exemplo aparece primeiro
        vmax_a = np.abs(PRE).max()
        saida = grade_numeros(PRE, lambda v: cor_ativacao(v, vmax_a)).move_to(RIGHT * 4.75 + DOWN * 0.35)
        moldura_saida = VGroup(*[Square(LADO).set_stroke(COR_SUAVE, 1, opacity=0.5) for _ in range(H * W)]) \
            .arrange_in_grid(rows=H, cols=W, buff=0).move_to(saida)
        rot_s = texto("mapa de ativação (saída do filtro)", 16, COR_SUAVE).next_to(saida, DOWN, buff=0.12)
        self.play(FadeIn(moldura_saida), FadeIn(rot_s))
        self.play(TransformFromCopy(jan, saida[r0 * W + c0]))
        self.next_slide()

        # Padding "same": nas bordas, a janela sai da imagem e encontra zeros
        nota_pad = texto("Nas bordas a janela passa da imagem: os pixels de fora contam como 0 (padding).\n"
                         "Por isso a saída tem o mesmo tamanho da entrada: 5×6.", 17, COR_SUAVE, line_spacing=1.1) \
            .to_edge(DOWN, buff=0.2)
        self.play(FadeOut(VGroup(conta, expl, rot_img)), FadeIn(borda), FadeIn(zeros), FadeIn(nota_pad),
                  Transform(jan, janela(0, 0)))
        self.play(FadeIn(saida[0]), run_time=0.3)

        # A janela desliza por todas as posições e o mapa vai sendo preenchido
        for r in range(H):
            for c in range(W):
                if (r, c) in [(0, 0), (r0, c0)]:
                    if (r, c) == (r0, c0):
                        self.play(Transform(jan, janela(r, c)), run_time=0.12)
                    continue
                self.play(Transform(jan, janela(r, c)), FadeIn(saida[r * W + c]), run_time=0.12)
        self.next_slide()

        self.mapa_pre = saida
        self.rot_mapa = rot_s
        self.play(FadeOut(VGroup(f, img, borda, zeros, filtro, rot_f, jan, nota_pad, moldura_saida)))

    # =================================================================
    def slide_5_relu_e_filtros(self):
        self.trocar_cabecalho("ReLU: os valores negativos viram zero")
        f = formula(r"\text{ReLU}(t) = \max(0,\, t)", tamanho=44).next_to(self.titulo, DOWN, buff=0.35)
        self.play(Write(f), self.mapa_pre.animate.move_to(LEFT * 3.2 + DOWN * 0.6),
                  self.rot_mapa.animate.move_to(LEFT * 3.2 + DOWN * 2.45))

        vmax = np.abs(PRE).max()
        relu = grade_numeros(RELU, lambda v: cor_ativacao(v, vmax)).move_to(RIGHT * 3.2 + DOWN * 0.6)
        rot_r = texto("depois do ReLU", 16, COR_SUAVE).move_to(RIGHT * 3.2 + DOWN * 2.45)
        seta = Arrow(self.mapa_pre.get_right(), relu.get_left(), buff=0.3, color=COR_SUAVE)
        negativos = [i for i in range(H * W) if PRE.ravel()[i] < 0]
        self.play(*[Indicate(self.mapa_pre[i], color=COR_RUIM, scale_factor=1.25) for i in negativos])
        self.play(GrowArrow(seta), TransformFromCopy(self.mapa_pre, relu), FadeIn(rot_r))
        nota = texto(f"{len(negativos)} valores negativos foram zerados. É isso que torna a rede não linear.",
                     20, COR_SUAVE).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(nota))
        self.next_slide()

        # 32 filtros diferentes -> 32 mapas (mostramos 6 reais)
        self.play(FadeOut(VGroup(f, self.mapa_pre, self.rot_mapa, seta, nota, rot_r)), relu.animate.set_opacity(0))
        self.remove(relu)
        self.trocar_cabecalho("32 filtros diferentes produzem 32 mapas")
        mapas = VGroup()
        for chave, m in L["mapas_relu"].items():
            m = np.array(m)
            g = VGroup(*[Square(0.27).set_fill(cor_ativacao(v, max(m.max(), 1e-6)), 1).set_stroke("#3A3F4A", 1)
                         for v in m.ravel()]).arrange_in_grid(rows=H, cols=W, buff=0)
            mapas.add(VGroup(g, texto(f"filtro {int(chave) + 1}", 15, COR_SUAVE)).arrange(DOWN, buff=0.1))
        mapas.arrange_in_grid(rows=2, cols=3, buff=(0.8, 0.45)).move_to(DOWN * 0.3)
        self.play(LaggedStart(*[FadeIn(m, scale=0.8) for m in mapas], lag_ratio=0.15))
        nota2 = texto("Cada filtro aprende a destacar uma combinação diferente de features vizinhas.\n"
                      "Saída da camada: 5 × 6 × 32. Aqui, 6 dos 32 mapas do mesmo paciente.", 19, COR_SUAVE,
                      line_spacing=1.1).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(nota2))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_6_max_pooling(self):
        self.trocar_cabecalho("Max-pooling: resume cada janela 2×2 pelo maior valor")
        f = formula(r"m_k[r, c] = \max_{0 \le u, v \le 1} \; a_k[2r+u,\; 2c+v]", tamanho=36) \
            .next_to(self.titulo, DOWN, buff=0.3)
        vmax = np.abs(PRE).max()
        mapa = grade_numeros(RELU, lambda v: cor_ativacao(v, vmax)).move_to(LEFT * 3.2 + DOWN * 0.8)
        pool = grade_numeros(POOL, lambda v: cor_ativacao(v, vmax), lado=0.8, tamanho=18).move_to(RIGHT * 3.4 + DOWN * 0.8)
        moldura = VGroup(*[Square(0.8).set_stroke(COR_SUAVE, 1, opacity=0.5) for _ in range(9)]) \
            .arrange_in_grid(rows=3, cols=3, buff=0).move_to(pool)
        rots = VGroup(texto("mapa 5×6 (depois do ReLU)", 16, COR_SUAVE).next_to(mapa, DOWN, buff=0.15),
                      texto("resultado 3×3", 16, COR_SUAVE).next_to(pool, DOWN, buff=0.15))
        self.play(Write(f), FadeIn(mapa), FadeIn(moldura), FadeIn(rots))

        def janela(r, c):
            """Janela 2x2 do pooling; na última linha (r = 2) só existe 1 linha de pixels."""
            linhas = [rr for rr in (2 * r, 2 * r + 1) if rr < H]
            cels = VGroup(*[mapa[rr * W + cc] for rr in linhas for cc in (2 * c, 2 * c + 1)])
            return SurroundingRectangle(cels, color=COR_DESTAQUE, buff=0, stroke_width=5)

        jan = janela(0, 0)
        self.play(Create(jan))
        for r in range(3):
            for c in range(3):
                linhas = [rr for rr in (2 * r, 2 * r + 1) if rr < H]
                bloco = RELU[linhas][:, [2 * c, 2 * c + 1]]
                pos = np.unravel_index(np.argmax(bloco), bloco.shape)
                maior = mapa[linhas[pos[0]] * W + 2 * c + pos[1]]
                self.play(Transform(jan, janela(r, c)), run_time=0.3)
                self.play(TransformFromCopy(maior, pool[r * 3 + c]), run_time=0.45)
                if (r, c) == (0, 0):
                    self.next_slide()
        nota = texto("A imagem encolhe de 5×6 para 3×3. Na última linha a janela só pega uma linha de pixels\n"
                     "(5 não é par). Os 32 mapas juntos: 3 × 3 × 32 = 288 números.", 18, COR_SUAVE,
                     line_spacing=1.1).to_edge(DOWN, buff=0.3)
        self.play(FadeOut(jan), FadeIn(nota))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_7_densas_e_sigmoide(self):
        self.trocar_cabecalho("De 288 números a uma probabilidade")
        f = formula(r"h = \text{ReLU}(W_1\, m + b_1)", r"\qquad", r"p = \sigma(w_2^{\top} h + b_2)", r"\qquad",
                    r"\sigma(t) = \frac{1}{1 + e^{-t}}", tamanho=36).next_to(self.titulo, DOWN, buff=0.3)
        self.play(Write(f))

        # Esquema: vetor achatado -> 16 neurônios -> 1 saída
        vetor = VGroup(*[Square(0.11).set_fill(COR_DESTAQUE, 0.25 + 0.75 * (i * 7 % 10) / 10).set_stroke(width=0)
                         for i in range(26)]).arrange(DOWN, buff=0.02).move_to(LEFT * 5.6 + DOWN * 1.2)
        rot_v = texto("m\n288 valores", 15, COR_SUAVE).next_to(vetor, LEFT, buff=0.15)
        oculta_vals = np.array(L["paciente"]["oculta"])
        neuronios = VGroup(*[
            Circle(radius=0.1, stroke_color=COR_BENIGNO, stroke_width=2)
            .set_fill(COR_BENIGNO, float(np.clip(v / max(oculta_vals.max(), 1e-6), 0, 1)))
            for v in oculta_vals
        ]).arrange(DOWN, buff=0.065).move_to(LEFT * 3.6 + DOWN * 1.2)
        rot_h = texto("h\n16 neurônios", 15, COR_BENIGNO).next_to(neuronios, DOWN, buff=0.12)
        saida = Circle(radius=0.22, stroke_color=COR_MALIGNO, stroke_width=3).set_fill(COR_MALIGNO, 0.9) \
            .move_to(LEFT * 1.9 + DOWN * 1.2)
        rot_o = texto("p", 20, COR_MALIGNO).next_to(saida, DOWN, buff=0.12)
        lig1 = VGroup(*[Line(vetor[i].get_right(), n.get_left(), stroke_width=0.6, color=COR_SUAVE).set_opacity(0.35)
                        for i in range(0, 26, 3) for n in neuronios[::2]])
        lig2 = VGroup(*[Line(n.get_right(), saida.get_left(), stroke_width=1, color=COR_SUAVE).set_opacity(0.6)
                        for n in neuronios])
        self.play(FadeIn(vetor), FadeIn(rot_v))
        self.play(Create(lig1), FadeIn(neuronios), FadeIn(rot_h))
        self.play(Create(lig2), FadeIn(saida), FadeIn(rot_o))
        self.next_slide()

        # A sigmoide, com os dois pacientes reais
        eixos = Axes(x_range=[-15, 15, 5], y_range=[0, 1, 0.5], x_length=6.2, y_length=2.8, tips=False,
                     axis_config={"color": COR_SUAVE, "include_numbers": True, "font_size": 18}) \
            .move_to(RIGHT * 3.3 + DOWN * 1.3)
        curva = eixos.plot(lambda t: 1 / (1 + np.exp(-t)), x_range=[-15, 15], color=COR_TEXTO, stroke_width=4)
        rot_x = texto("t = w₂ᵀh + b₂  (qualquer número real)", 15, COR_SUAVE).next_to(eixos, DOWN, buff=0.25).shift(LEFT * 1.2)
        rot_y = texto("p = σ(t)", 16, COR_SUAVE).next_to(eixos, UP, buff=0.1).align_to(eixos, LEFT)
        meio = DashedLine(eixos.c2p(-15, 0.5), eixos.c2p(15, 0.5), color=COR_SUAVE, stroke_width=1.5)
        self.play(Create(eixos), Create(curva), FadeIn(rot_x), FadeIn(rot_y), Create(meio))

        tm, pm = L["paciente"]["logit"], L["paciente"]["prob"]
        tb, pb = L["benigno"]["logit"], L["benigno"]["prob"]
        ponto_m = Dot(eixos.c2p(np.clip(tm, -15, 15), pm), color=COR_MALIGNO, radius=0.11)
        ponto_b = Dot(eixos.c2p(np.clip(tb, -15, 15), pb), color=COR_BENIGNO, radius=0.11)
        rot_m = texto(f"paciente maligno\nt = {br(tm, 1)}  →  p = {br(pm, 4)}", 15, COR_MALIGNO, line_spacing=1.0) \
            .next_to(ponto_m, DOWN, buff=0.15).shift(LEFT * 0.9)
        rot_b = texto(f"paciente benigno\nt = {br(tb, 1)}  →  p = {br(pb, 4)}", 15, COR_BENIGNO, line_spacing=1.0) \
            .next_to(ponto_b, UP, buff=0.15)
        self.play(FadeIn(ponto_m, scale=2), FadeIn(rot_m))
        self.play(FadeIn(ponto_b, scale=2), FadeIn(rot_b))
        regra = texto("Regra de decisão: maligno se p > 0,5", 22, COR_DESTAQUE).next_to(eixos, UP, buff=0.12).align_to(eixos, RIGHT)
        self.play(FadeIn(regra))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_8_perda(self):
        self.trocar_cabecalho("Como a rede sabe que errou: entropia cruzada")
        f = formula(r"\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n} \left[\, y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \,\right]",
                    tamanho=40).next_to(self.titulo, DOWN, buff=0.35)
        self.play(Write(f))

        eixos = Axes(x_range=[0, 1, 0.25], y_range=[0, 5, 1], x_length=5.6, y_length=3.3, tips=False,
                     axis_config={"color": COR_SUAVE, "include_numbers": True, "font_size": 18}) \
            .move_to(LEFT * 3.2 + DOWN * 1.2)
        curva = eixos.plot(lambda p: -np.log(p), x_range=[0.007, 1], color=COR_MALIGNO, stroke_width=4)
        rot_x = texto("p previsto", 15, COR_SUAVE).next_to(eixos.x_axis, RIGHT, buff=0.15)
        rot_y = texto("perda = −ln p, para um paciente MALIGNO (y = 1)", 16, COR_MALIGNO) \
            .next_to(eixos, UP, buff=0.05).align_to(eixos, LEFT)
        self.play(Create(eixos), Create(curva), FadeIn(rot_x), FadeIn(rot_y))

        exemplos = VGroup()
        for p_ex, cor, comentario in [(0.9, COR_BOM, "confiante e certo"), (0.5, COR_DESTAQUE, "em dúvida"),
                                      (0.1, COR_RUIM, "confiante e ERRADO")]:
            ponto = Dot(eixos.c2p(p_ex, -np.log(p_ex)), color=cor, radius=0.1)
            linha = VGroup(formula(rf"p = {tex(p_ex, 1)} \;\Rightarrow\; -\ln {tex(p_ex, 1)} =", tex(-np.log(p_ex), 3),
                                   tamanho=32), texto(comentario, 18, cor))
            linha[0][1].set_color(cor)
            linha.arrange(RIGHT, buff=0.3)
            exemplos.add(VGroup(ponto, linha))
        VGroup(*[e[1] for e in exemplos]).arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to(RIGHT * 3.35 + DOWN * 1.2)
        for e in exemplos:
            self.play(FadeIn(e[0], scale=2), FadeIn(e[1]))
        nota = texto("Errar com muita confiança custa caro. O treino ajusta os pesos para diminuir essa perda.",
                     19, COR_SUAVE).to_edge(DOWN, buff=0.15)
        self.play(FadeIn(nota))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_9_adam_e_treino(self):
        self.trocar_cabecalho("Adam: como os pesos são ajustados")
        linhas = VGroup(
            formula(r"m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t", r"\qquad", r"v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2",
                    tamanho=34),
            formula(r"\hat{m}_t = \frac{m_t}{1-\beta_1^t}", r"\qquad", r"\hat{v}_t = \frac{v_t}{1-\beta_2^t}", r"\qquad",
                    r"\theta_t = \theta_{t-1} - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}", tamanho=34),
        ).arrange(DOWN, buff=0.4).move_to(UP * 1.2)
        legenda = VGroup(
            texto("g = gradiente da perda em relação ao peso θ (a direção em que a perda cresce)", 18, COR_SUAVE),
            texto("m = média móvel do gradiente (direção)  ·  v = média móvel do gradiente² (tamanho do passo)", 18, COR_SUAVE),
            texto("β₁ = 0,9   ·   β₂ = 0,999   ·   ε = 10⁻⁷   ·   η escolhida por grid: 0,1 · 0,01 · 0,001 · 0,0001",
                  19, COR_DESTAQUE),
            texto("lotes de 32 pacientes  ·  100 épocas (passadas completas pelo treino)", 19, COR_DESTAQUE),
        ).arrange(DOWN, buff=0.18).move_to(DOWN * 1.7)
        self.play(Write(linhas[0]))
        self.play(Write(linhas[1]))
        self.play(FadeIn(legenda))
        self.next_slide()

        # Curva de treino REAL
        self.limpar_tudo()
        self.trocar_cabecalho("O treino real: a perda caindo ao longo de 100 épocas")
        hist = L["historico_loss"]
        eixos = Axes(x_range=[0, 100, 20], y_range=[0, 0.6, 0.1], x_length=9, y_length=4.2, tips=False,
                     axis_config={"color": COR_SUAVE, "include_numbers": True, "font_size": 20}) \
            .move_to(DOWN * 0.3)
        rot_x = texto("época", 18, COR_SUAVE).next_to(eixos, DOWN, buff=0.3)
        rot_y = texto("perda no treino (entropia cruzada)", 18, COR_SUAVE).next_to(eixos, UP, buff=0.1).align_to(eixos, LEFT)
        pts = [eixos.c2p(i + 1, v) for i, v in enumerate(hist)]
        curva = VMobject(stroke_color=COR_DESTAQUE, stroke_width=4).set_points_as_corners(pts)
        self.play(Create(eixos), FadeIn(rot_x), FadeIn(rot_y))
        self.play(Create(curva), run_time=4, rate_func=linear)
        marcos = VGroup()
        for ep in [1, 10, 50, 100]:
            p = eixos.c2p(ep, hist[ep - 1])
            marcos.add(VGroup(Dot(p, color=COR_TEXTO, radius=0.06),
                              texto(f"época {ep}: {br(hist[ep - 1], 3)}", 16, COR_TEXTO)
                              .next_to(p, RIGHT if ep == 1 else UR, buff=0.12 if ep == 1 else 0.08)))
        self.play(LaggedStart(*[FadeIn(m) for m in marcos], lag_ratio=0.3))
        nota = texto("η = 0,001 · imagens 5×6 · 455 pacientes de treino", 18, COR_SUAVE).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(nota))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_10_teste(self):
        self.trocar_cabecalho("A rede treinada nos 114 pacientes do teste")
        t = L["teste"]
        probs, y = np.array(t["prob"]), np.array(t["y"])
        eixo = NumberLine(x_range=[0, 1, 0.25], length=10, color=COR_SUAVE, include_numbers=True, font_size=22,
                          decimal_number_config={"num_decimal_places": 2}).move_to(UP * 0.35)
        rng = np.random.default_rng(0)
        pontos = VGroup(*[
            Dot([eixo.n2p(p)[0], eixo.get_y() + (0.55 if yi == 0 else 1.25) + rng.uniform(-0.25, 0.25), 0],
                radius=0.05, color=COR_BENIGNO if yi == 0 else COR_MALIGNO, fill_opacity=0.8)
            for p, yi in zip(probs, y)
        ])
        corte = DashedLine([eixo.n2p(0.5)[0], eixo.get_y() - 0.3, 0], [eixo.n2p(0.5)[0], eixo.get_y() + 1.75, 0],
                           color=COR_DESTAQUE, stroke_width=3)
        rot_c = texto("corte: 0,5", 16, COR_DESTAQUE).next_to(corte, UP, buff=0.05)
        rot_x = texto("probabilidade prevista de ser maligno", 16, COR_SUAVE).next_to(eixo, DOWN, buff=0.45)
        rots = VGroup(texto("benignos", 16, COR_BENIGNO).move_to([-6.1, eixo.get_y() + 0.55, 0]),
                      texto("malignos", 16, COR_MALIGNO).move_to([-6.1, eixo.get_y() + 1.25, 0]))
        self.play(Create(eixo), FadeIn(rot_x), FadeIn(rots))
        self.play(LaggedStart(*[FadeIn(p, scale=0.3) for p in pontos], lag_ratio=0.01), run_time=2)
        self.play(Create(corte), FadeIn(rot_c))
        self.next_slide()

        # Matriz de confusão e métricas
        tp, fp, tn, fn = t["tp"], t["fp"], t["tn"], t["fn"]

        def cel(valor, nome, cor):
            r = Rectangle(width=2.0, height=0.85, stroke_color=COR_SUAVE, stroke_width=1.5).set_fill(cor, 0.25)
            return VGroup(r, VGroup(texto(str(valor), 26, COR_TEXTO, weight=BOLD), texto(nome, 13, COR_SUAVE))
                          .arrange(DOWN, buff=0.04).move_to(r))

        matriz = VGroup(cel(tp, "maligno → maligno (VP)", COR_BOM), cel(fn, "maligno → benigno (FN)", COR_RUIM),
                        cel(fp, "benigno → maligno (FP)", COR_RUIM), cel(tn, "benigno → benigno (VN)", COR_BOM)) \
            .arrange_in_grid(rows=2, cols=2, buff=0).move_to(LEFT * 3.9 + DOWN * 2.35)
        sens, espec = tp / (tp + fn), tn / (tn + fp)
        metricas = VGroup(
            formula(rf"\text{{Sensibilidade}} = \frac{{VP}}{{VP + FN}} = \frac{{{tp}}}{{{tp + fn}}} =", tex(sens, 3), tamanho=28),
            formula(rf"\text{{Especificidade}} = \frac{{VN}}{{VN + FP}} = \frac{{{tn}}}{{{tn + fp}}} =", tex(espec, 3), tamanho=28),
            formula(r"\text{Acurácia balanceada} = \frac{\text{Sens.} + \text{Espec.}}{2} =", tex((sens + espec) / 2, 3), tamanho=28),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT).move_to(RIGHT * 2.9 + DOWN * 2.35)
        for m in metricas:
            m[1].set_color(COR_DESTAQUE)
        self.play(FadeIn(matriz))
        for m in metricas:
            self.play(Write(m), run_time=0.9)
        self.next_slide()

        aviso = texto("Atenção: isto é UM treino num teste de 114 pacientes. A comparação justa usa os 10 folds (próxima etapa).",
                      17, COR_RUIM).to_edge(DOWN, buff=0.12)
        self.play(FadeIn(aviso))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_11_saida(self):
        self.trocar_cabecalho("Saída do LEARN")
        itens = VGroup(
            texto("1.  Para cada paciente, a probabilidade p de ser maligno", 26),
            texto("2.  A classe prevista: maligno se p > 0,5", 26),
            texto("3.  Uma rede treinada, pronta para pacientes novos", 26),
            texto("Próxima etapa → AVALIAÇÃO: isso é melhor que os métodos tradicionais?", 26, COR_DESTAQUE),
        ).arrange(DOWN, buff=0.5, aligned_edge=LEFT).move_to(DOWN * 0.2)
        itens[3].shift(DOWN * 0.3)
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.3) for i in itens], lag_ratio=0.4), run_time=2.5)
        self.next_slide()
