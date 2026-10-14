"""
CENA 3 — Etapa OPTIMIZE: formato da imagem e seleção de features
=================================================================

Roteiro (cada método slide_... termina num `self.next_slide()`, onde a apresentação pausa):
   1. Título
   2. O que entra: os 30 valores na ordem do Map, e a pergunta (formato? quais features?)
   3. A solução é um vetor: x̂ = (h, w, b1, ..., b30)
   4. As restrições (quando uma solução é válida)
   5. A função objetivo (Eq. 1): comparar imagens de pacientes consecutivos
   6. Decodificação: de um vetor qualquer a uma solução válida (exemplo real)
   7. PSO: a fórmula e o que cada termo faz
   8. PSO: o enxame real se movendo + convergência
   9. EPSO: fórmulas e enxame real
  10. ABC: fórmulas e colônia real
  11. Comparação dos três: formato, f, acurácia e as imagens geradas
  12. O que a Eq. 1 premia: poucas features (e ignora o arranjo dos pixels)
  13. Saída da etapa e próxima etapa

COMO RODAR (de dentro da pasta apresentacao/):
    .venv/bin/manim-slides render -ql cena3_optimize.py Cena3Optimize   # prévia rápida
    .venv/bin/manim-slides render cena3_optimize.py Cena3Optimize       # 1080p
    .venv/bin/manim-slides present Cena3Optimize                        # apresentar

Todos os números e trajetórias são REAIS (dados/mol_dados.json, chave "optimize"):
os enxames foram rodados com o mesmo código, sementes e orçamento do experimento.
"""
import numpy as np
from manim import *

from estilo import (COR_BENIGNO, COR_BOM, COR_DESTAQUE, COR_FUNDO, COR_MALIGNO, COR_RUIM,
                    COR_SUAVE, COR_TEXTO, DADOS, N_FEATURES, CenaBase, br, formula,
                    nome_curto, pixel, tex, texto)

OPT = DADOS["optimize"]
NOMES = DADOS["feature_names"]
ORDEM = DADOS["ordem"]                     # ORDEM[k] = feature (0..29) na posição k do ranking
M = N_FEATURES

# Cores para o valor da função objetivo f (baixo = amarelo, alto = roxo)
F_MIN, F_MAX = 0.75, 1.6
COR_F_BAIXO, COR_F_ALTO = "#F2C94C", "#5B3E96"

# Em que avaliação termina cada "foto" (snapshot) da população de cada otimizador
AVALIACAO_DO_SNAPSHOT = {
    "pso": lambda k: 50 * (k + 1),         # 50 iniciais + 50 por iteração
    "epso": lambda k: 50 + 100 * k,        # 2 avaliações por partícula (original + réplica)
    "abc": lambda k: 25 + 50 * k,          # 25 fontes + 25 empregadas + 25 observadoras
}


def cor_f(f):
    t = float(np.clip((f - F_MIN) / (F_MAX - F_MIN), 0, 1))
    return interpolate_color(ManimColor(COR_F_BAIXO), ManimColor(COR_F_ALTO), t)


def grade(valores, h, w, lado):
    """Imagem h x w preenchida linha a linha com os valores (z-scores) dados."""
    g = VGroup(*[pixel(v, lado) for v in valores[: h * w]])
    return g.arrange_in_grid(rows=h, cols=w, buff=0)


def caixa_num(conteudo, largura=0.36, cor=COR_TEXTO, tamanho=18, borda=COR_SUAVE):
    """Uma caixinha com um número dentro (usada para o vetor x̂)."""
    r = Rectangle(width=largura, height=0.42, stroke_color=borda, stroke_width=1.5)
    t = texto(conteudo, tamanho, cor)
    if t.width > largura - 0.06:
        t.scale_to_fit_width(largura - 0.06)
    return VGroup(r, t.move_to(r))


class Cena3Optimize(CenaBase):
    def construct(self):
        self.slide_1_titulo()
        self.slide_2_entrada()
        self.slide_3_vetor()
        self.slide_4_restricoes()
        self.slide_5_funcao_objetivo()
        self.slide_6_decodificacao()
        self.slide_7_pso_formula()
        self.slide_8_pso_enxame()
        self.slide_9_epso()
        self.slide_10_abc()
        self.slide_11_comparacao()
        self.slide_12_o_que_a_eq1_premia()
        self.slide_13_saida()

    # =================================================================
    # Ajudantes
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
        etapa = texto("ETAPA 2", 30, COR_DESTAQUE)
        titulo = texto("OPTIMIZE", 96, weight=BOLD)
        sub = texto("Qual formato e quais features entram na imagem?", 32, COR_SUAVE)
        VGroup(etapa, titulo, sub).arrange(DOWN, buff=0.35)
        self.play(FadeIn(etapa, shift=DOWN * 0.3))
        self.play(Write(titulo))
        self.play(FadeIn(sub, shift=UP * 0.3))
        self.next_slide()
        self.limpar()

    # =================================================================
    def slide_2_entrada(self):
        self.trocar_cabecalho("O que entra: os 30 valores na ordem do Map")
        z = OPT["pacientes_ordenados"]["exemplo"]       # z-scores já na ordem do ranking
        lado = 0.36
        self.faixa = VGroup(*[pixel(v, lado) for v in z]).arrange(RIGHT, buff=0).move_to(UP * 2.0)
        self.posicoes = VGroup(*[texto(str(k + 1), 12, COR_DESTAQUE).next_to(self.faixa[k], UP, buff=0.08)
                                 for k in range(M)])
        rot = texto("posição no ranking (1 = feature mais importante)", 18, COR_SUAVE) \
            .next_to(self.posicoes, UP, buff=0.12)
        self.play(LaggedStart(*[FadeIn(p) for p in self.faixa], lag_ratio=0.03), FadeIn(self.posicoes), FadeIn(rot))
        self.next_slide()

        # Algumas imagens possíveis com as primeiras k features
        candidatas = VGroup()
        for h, w in [(5, 6), (4, 4), (3, 3), (1, 2)]:
            img = grade(z, h, w, 0.3)
            r = VGroup(img, texto(f"{h}×{w} = {h * w} features", 18, COR_SUAVE).next_to(img, DOWN, buff=0.15))
            candidatas.add(r)
        candidatas.arrange(RIGHT, buff=0.9, aligned_edge=DOWN).move_to(DOWN * 0.6)
        self.play(LaggedStart(*[TransformFromCopy(self.faixa, c[0]) for c in candidatas], lag_ratio=0.3),
                  FadeIn(VGroup(*[c[1] for c in candidatas])), run_time=2.5)

        perguntas = VGroup(
            texto("Quantos pixels?  Em que formato?  Quais features?", 26, COR_DESTAQUE),
            texto("Cada feature entra ou não: 2³⁰ ≈ 1 bilhão de subconjuntos, vezes os formatos possíveis", 20, COR_SUAVE),
            texto("→ grande demais para testar tudo: o MOL usa uma busca por enxame (metaheurística)", 20, COR_SUAVE),
        ).arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(perguntas))
        self.next_slide()
        self.play(FadeOut(VGroup(candidatas, perguntas, rot)))

    # =================================================================
    def slide_3_vetor(self):
        self.trocar_cabecalho("Cada solução candidata é um vetor com 32 números")
        f = formula(r"\hat{x} = (\,h,\; w,\; b_1,\; b_2,\; \dots,\; b_{30}\,)", r"\qquad b_k \in \{0, 1\}",
                    tamanho=44).move_to(DOWN * 0.1)
        self.play(Write(f))
        self.next_slide()

        # Exemplo: h = 4, w = 4 e os 16 bits da solução encontrada pelo ABC
        mascara = OPT["otimizadores"]["abc"]["mascara"]
        bits = VGroup(*[caixa_num(str(b), 0.36, COR_DESTAQUE if b else COR_SUAVE) for b in mascara]) \
            .arrange(RIGHT, buff=0)
        for k, c in enumerate(bits):
            c.move_to([self.faixa[k].get_x(), 0.95, 0])
        h_box = caixa_num("4", 0.5, COR_BOM, 22, COR_BOM)
        w_box = caixa_num("4", 0.5, COR_BOM, 22, COR_BOM)
        VGroup(h_box, w_box).arrange(RIGHT, buff=0.05).next_to(bits, LEFT, buff=0.25)
        rot_hw = VGroup(texto("h", 18, COR_BOM).next_to(h_box, DOWN, buff=0.08),
                        texto("w", 18, COR_BOM).next_to(w_box, DOWN, buff=0.08))
        rot_b = texto("b_k: a feature da posição k entra (1) ou não (0)", 18, COR_DESTAQUE) \
            .next_to(bits, DOWN, buff=0.12)
        self.play(f.animate.scale(0.75).to_edge(DOWN, buff=0.4))
        self.play(FadeIn(h_box), FadeIn(w_box), FadeIn(rot_hw),
                  LaggedStart(*[FadeIn(b) for b in bits], lag_ratio=0.03), FadeIn(rot_b))
        self.next_slide()

        # As features marcadas acendem e montam a imagem 4 x 4, na ordem do ranking
        escolhidas = [k for k, b in enumerate(mascara) if b]
        alvo = VGroup(*[Square(0.42) for _ in escolhidas]).arrange_in_grid(rows=4, cols=4, buff=0) \
            .move_to(DOWN * 1.6)
        self.play(*[self.faixa[k].animate.set_opacity(0.2) for k in range(M) if not mascara[k]])
        copias = [self.faixa[k].copy() for k in escolhidas]
        self.play(LaggedStart(*[c.animate.scale(0.42 / 0.36).move_to(alvo[i]) for i, c in enumerate(copias)],
                              lag_ratio=0.08), run_time=2.5)
        nota = texto("h × w = 4 × 4 = 16 pixels  ←  16 features marcadas, na ordem do Map", 22, COR_TEXTO) \
            .next_to(alvo, DOWN, buff=0.3)
        self.play(FadeOut(f), FadeIn(nota))
        self.next_slide()

        self.vetor = VGroup(h_box, w_box, rot_hw, bits, rot_b)
        self.play(FadeOut(VGroup(*copias, nota, self.vetor)), self.faixa.animate.set_opacity(1))

    # =================================================================
    def slide_4_restricoes(self):
        self.trocar_cabecalho("Quando uma solução é válida? As restrições")
        f = formula(r"2 \;\le\; h \cdot w \;\le\; M", r"\qquad\qquad", r"h \cdot w \;=\; \sum_{k=1}^{M} b_k",
                    tamanho=42).move_to(UP * 0.6)
        expl = VGroup(
            texto("pelo menos 2 pixels e no máximo 30 (o número de features)", 18, COR_SUAVE),
            texto("cada pixel recebe exatamente uma feature: sem preencher com zeros", 18, COR_SUAVE),
        )
        expl[0].next_to(f[0], DOWN, buff=0.3)
        expl[1].next_to(f[2], DOWN, buff=0.3)
        self.play(FadeOut(VGroup(self.faixa, self.posicoes)), Write(f))
        self.play(FadeIn(expl))
        self.next_slide()

        z = OPT["pacientes_ordenados"]["exemplo"]

        def cartao(img, titulo_txt, ok):
            simbolo = texto("✓ válida" if ok else "✗ inválida", 22, COR_BOM if ok else COR_RUIM)
            t = texto(titulo_txt, 17, COR_SUAVE)
            return VGroup(img, t, simbolo).arrange(DOWN, buff=0.15)

        # 1) 4x4 com 16 features: válido
        c1 = cartao(grade(z, 4, 4, 0.3), "4×4 = 16 pixels, 16 marcadas", True)
        # 2) 4x4 com só 13 features: sobram pixels vazios
        img2 = grade(z, 4, 4, 0.3)
        for k in range(13, 16):
            img2[k].set_fill(COR_FUNDO, 1).set_stroke(COR_RUIM, 2)
        c2 = cartao(img2, "4×4 = 16 pixels, só 13 marcadas", False)
        # 3) 6x6 = 36 pixels: mais pixels que features
        img3 = VGroup(*[Square(0.2, stroke_color=COR_RUIM if k >= 30 else COR_SUAVE, stroke_width=1)
                        for k in range(36)]).arrange_in_grid(rows=6, cols=6, buff=0)
        c3 = cartao(img3, "6×6 = 36 pixels > 30 features", False)
        cartoes = VGroup(c1, c2, c3).arrange(RIGHT, buff=1.4, aligned_edge=DOWN).move_to(DOWN * 2.0)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cartoes], lag_ratio=0.4))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_5_funcao_objetivo(self):
        self.trocar_cabecalho("O que o otimizador minimiza: a Eq. 1 do artigo")
        f = formula(r"f(\hat{x}) = \frac{1}{N}\sum_{i=1}^{N-1}", r"\frac{1}{SM}\sum_{j=1}^{SM}\left(X_{i,j} - X_{i+1,j}\right)^2",
                    tamanho=46).move_to(UP * 1.6)
        chave = Brace(f[1], DOWN, color=COR_DESTAQUE)
        rot = texto("diferença média entre a imagem do paciente i e a do paciente seguinte (i + 1)",
                    20, COR_DESTAQUE).next_to(chave, DOWN, buff=0.1)
        nota = texto("SM = h · w = número de pixels · minimizar f = deixar imagens vizinhas parecidas",
                     18, COR_SUAVE).next_to(rot, DOWN, buff=0.25)
        self.play(Write(f))
        self.play(GrowFromCenter(chave), FadeIn(rot))
        self.play(FadeIn(nota))
        self.next_slide()

        # Sequência real de 6 pacientes consecutivos (ordenados por classe), formato 4x4
        seq = OPT["sequencia"]
        imgs = VGroup()
        for zi, yi in zip(seq["z"], seq["y"]):
            g = grade(zi, 4, 4, 0.26)
            g.add(SurroundingRectangle(g, color=COR_BENIGNO if yi == 0 else COR_MALIGNO, buff=0.03, stroke_width=3))
            imgs.add(g)
        imgs.arrange(RIGHT, buff=0.85).move_to(DOWN * 1.3)
        mses = VGroup()
        for k, v in enumerate(seq["mse"]):
            pos = (imgs[k].get_right() + imgs[k + 1].get_left()) / 2
            mses.add(texto(br(v, 2), 18, COR_TEXTO).move_to(pos + UP * 0.05))
            if seq["y"][k] != seq["y"][k + 1]:
                mses.add(texto("B→M", 14, COR_SUAVE).move_to(pos + UP * 0.45))
        rot_ordem = texto("Ordem das amostras (o artigo não diz): ordenadas por classe, benignos e depois malignos",
                          18, COR_SUAVE).to_edge(DOWN, buff=0.75)
        fronteira = texto("Só 1 dos 454 pares cruza a fronteira. Repare: os malignos variam muito mais entre si (3,25 e 2,42).",
                          18, COR_DESTAQUE).to_edge(DOWN, buff=0.35)
        self.play(FadeOut(nota), LaggedStart(*[FadeIn(i) for i in imgs], lag_ratio=0.15), FadeIn(rot_ordem))
        self.play(LaggedStart(*[FadeIn(m) for m in mses], lag_ratio=0.2))
        self.play(FadeIn(fronteira))
        self.next_slide()

        # Zoom num par: imagem i, imagem i + 1 e a diferença ao quadrado pixel a pixel
        self.play(FadeOut(VGroup(imgs, mses, rot_ordem, fronteira)))
        za, zb = np.array(seq["z"][0][:16]), np.array(seq["z"][1][:16])
        ga, gb = grade(za, 4, 4, 0.5), grade(zb, 4, 4, 0.5)
        dif = (za - zb) ** 2
        gd = VGroup(*[Square(0.5).set_fill(interpolate_color(ManimColor(COR_FUNDO), ManimColor(COR_DESTAQUE),
                                                              min(d / 1.0, 1)), 1).set_stroke(COR_SUAVE, 1)
                      for d in dif]).arrange_in_grid(rows=4, cols=4, buff=0)
        menos = MathTex("-", font_size=60, color=COR_TEXTO)
        igual = MathTex(r"\xrightarrow{(\cdot)^2}", font_size=60, color=COR_TEXTO)
        resultado = formula(r"\text{média} =", tex(seq["mse"][0], 3), tamanho=40)
        resultado[1].set_color(COR_DESTAQUE)
        linha = VGroup(ga, menos, gb, igual, gd, resultado).arrange(RIGHT, buff=0.4).move_to(DOWN * 1.4)
        rots = VGroup(texto("paciente i", 18, COR_BENIGNO).next_to(ga, DOWN, buff=0.15),
                      texto("paciente i + 1", 18, COR_BENIGNO).next_to(gb, DOWN, buff=0.15),
                      texto("diferença²", 18, COR_DESTAQUE).next_to(gd, DOWN, buff=0.15))
        self.play(FadeIn(ga), FadeIn(gb), FadeIn(menos), FadeIn(rots[:2]))
        self.play(FadeIn(igual), TransformFromCopy(VGroup(ga, gb), gd), FadeIn(rots[2]))
        self.play(Write(resultado))
        self.next_slide()

        ff = OPT["f_formas"]
        valores = texto(f"f é a média sobre os 454 pares.  Exemplos reais:  5×6 (todas) = {br(ff['5x6'], 3)}  ·  "
                        f"4×4 (16 primeiras) = {br(ff['4x4'], 3)}  ·  melhor possível = {br(ff['otimo'], 3)}",
                        19, COR_TEXTO).to_edge(DOWN, buff=0.35)
        if valores.width > 13.6:
            valores.scale_to_fit_width(13.6)
        self.play(FadeIn(valores))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_6_decodificacao(self):
        self.trocar_cabecalho("Decodificação: de números quaisquer a uma solução válida")
        ex = OPT["exemplo_decodificacao"]
        x = ex["x"]
        b = np.array(x[2:])
        mascara_final = np.array(ex["mascara"])
        h, w = ex["forma"]
        k_alvo = h * w

        intro = texto("O enxame trabalha com números reais: h e w entre 1 e 30, cada b_k entre 0 e 1. "
                      "Exemplo REAL de uma partícula:", 19, COR_SUAVE).next_to(self.titulo, DOWN, buff=0.2)
        if intro.width > 13.4:
            intro.scale_to_fit_width(13.4)
        self.play(FadeIn(intro))

        # Passo 1: arredondar h e w
        hw = formula(rf"h = {tex(x[0], 3)} \;\rightarrow\; {ex['h_arred']}", r"\qquad",
                     rf"w = {tex(x[1], 3)} \;\rightarrow\; {ex['w_arred']}", r"\qquad",
                     rf"h \cdot w = {k_alvo} \le 30 \;\checkmark", tamanho=34).move_to(UP * 2.0)
        passo1 = texto("1. arredonda h e w (se h·w passar de 30, reduz w)", 19, COR_DESTAQUE) \
            .next_to(hw, DOWN, buff=0.15)
        self.play(Write(hw), FadeIn(passo1))
        self.next_slide()

        # Passo 2: os 30 valores b_k como barras, com a linha de corte 0,5
        base_y, escala, largura = -1.2, 2.0, 0.3
        x0 = -0.5 * M * (largura + 0.05)
        barras = VGroup()
        for k, v in enumerate(b):
            r = Rectangle(width=largura, height=max(v * escala, 0.01), stroke_width=0)
            r.set_fill(COR_DESTAQUE if v > 0.5 else COR_SUAVE, 0.9)
            r.move_to([x0 + k * (largura + 0.05), base_y, 0], aligned_edge=DOWN)
            barras.add(r)
        corte = DashedLine([x0 - 0.3, base_y + 0.5 * escala, 0], [x0 + M * (largura + 0.05), base_y + 0.5 * escala, 0],
                           color=COR_TEXTO)
        rot_corte = texto("0,5", 18).next_to(corte, LEFT, buff=0.1)
        rot_pos = VGroup(*[texto(str(k + 1), 11, COR_SUAVE).next_to(barras[k], DOWN, buff=0.06).set_y(base_y - 0.15)
                           for k in range(M)])
        n_marc = int((b > 0.5).sum())
        passo2 = texto(f"2. marca as features com b_k > 0,5  →  {n_marc} marcadas, mas o formato pede {k_alvo}",
                       19, COR_DESTAQUE).move_to(DOWN * 2.1)
        self.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in barras], lag_ratio=0.03), Create(corte),
                  FadeIn(rot_corte), FadeIn(rot_pos))
        self.play(FadeIn(passo2))
        self.next_slide()

        # Passo 3: completa com as de maior b_k entre as não marcadas
        adicionadas = [k for k in range(M) if mascara_final[k] and b[k] <= 0.5]
        removidas = [k for k in range(M) if not mascara_final[k] and b[k] > 0.5]
        if adicionadas:
            txt3 = f"3. faltam {len(adicionadas)}: entram as de MAIOR b_k entre as não marcadas"
        else:
            txt3 = f"3. sobram {len(removidas)}: saem as de MENOR b_k entre as marcadas"
        passo3 = texto(txt3, 19, COR_MALIGNO).next_to(passo2, DOWN, buff=0.15)
        self.play(FadeIn(passo3),
                  *[barras[k].animate.set_fill(COR_MALIGNO, 1) for k in adicionadas],
                  *[barras[k].animate.set_fill(COR_RUIM, 0.3) for k in removidas])
        self.next_slide()

        # Resultado: a imagem h x w com exatamente h*w features
        z = OPT["pacientes_ordenados"]["exemplo"]
        valores = [z[k] for k in range(M) if mascara_final[k]]
        img = grade(valores, h, w, 0.32)
        resultado = VGroup(img, texto(f"{h}×{w} com {k_alvo} features: válida", 20, COR_BOM)) \
            .arrange(DOWN, buff=0.15).to_corner(DR, buff=0.4)
        self.play(FadeOut(VGroup(passo2, passo3)), FadeIn(resultado))
        nota = texto("O artigo não diz como trata as restrições: esse reparo é uma escolha do projeto.",
                     18, COR_SUAVE).to_corner(DL, buff=0.45)
        self.play(FadeIn(nota))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_7_pso_formula(self):
        self.trocar_cabecalho("PSO · Particle Swarm Optimization")
        ideia = texto("50 partículas voam pelo espaço de soluções. Cada uma lembra o melhor lugar onde já esteve (p_i)\n"
                      "e conhece o melhor lugar já achado pelo enxame inteiro (g).", 20, COR_SUAVE, line_spacing=1.2) \
            .next_to(self.titulo, DOWN, buff=0.25)
        f_v = formula(r"v_i^{t+1} =", r"\omega\, v_i^{t}", "+", r"c_1 r_1 \left(p_i - x_i^{t}\right)", "+",
                      r"c_2 r_2 \left(g - x_i^{t}\right)", tamanho=44)
        f_x = formula(r"x_i^{t+1} = x_i^{t} + v_i^{t+1}", tamanho=40)
        VGroup(f_v, f_x).arrange(DOWN, buff=0.35).move_to(UP * 0.5)
        self.play(FadeIn(ideia), Write(f_v), Write(f_x))

        cores = [COR_SUAVE, COR_BENIGNO, COR_MALIGNO]
        partes = [f_v[1], f_v[3], f_v[5]]
        nomes = ["inércia", "memória", "cooperação"]
        explicacoes = ["inércia: continuar na mesma direção", "memória: voltar ao melhor lugar que ELA já visitou (p_i)",
                       "cooperação: ir para o melhor lugar que o ENXAME já achou (g)"]
        chaves = VGroup()
        for p, c, n in zip(partes, cores, nomes):
            p.set_color(c)
            ch = Brace(p, DOWN, color=c)
            chaves.add(VGroup(ch, texto(n, 17, c).next_to(ch, DOWN, buff=0.08)))
        legenda_termos = VGroup(*[texto(e, 18, c) for e, c in zip(explicacoes, cores)]) \
            .arrange(DOWN, buff=0.1, aligned_edge=LEFT).move_to(DOWN * 1.9)
        chaves.add(legenda_termos)
        # empurra o x para baixo das chaves
        self.play(f_x.animate.next_to(f_v, UP, buff=0.3), LaggedStart(*[FadeIn(c) for c in chaves], lag_ratio=0.3))
        params = texto("ω = 0,8   ·   c₁ = c₂ = 1,8   ·   r₁, r₂ sorteados entre 0 e 1   ·   500 avaliações de f",
                       19, COR_DESTAQUE).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(params))
        self.next_slide()

        # Esquema de UMA partícula: as três setas somam a nova velocidade
        self.play(FadeOut(VGroup(ideia, chaves, params)), VGroup(f_v, f_x).animate.scale(0.7).to_edge(LEFT, buff=0.4).shift(UP * 1.2))
        o = np.array([2.0, -1.4, 0])
        part = Dot(o, radius=0.1, color=COR_TEXTO)
        pi = Star(n=5, outer_radius=0.18, color=COR_BENIGNO, fill_opacity=1).move_to(o + np.array([-1.2, 1.8, 0]))
        gg = Star(n=5, outer_radius=0.22, color=COR_MALIGNO, fill_opacity=1).move_to(o + np.array([2.6, 1.6, 0]))
        rot_pi = texto("p_i", 18, COR_BENIGNO).next_to(pi, UP, buff=0.08)
        rot_g = texto("g", 18, COR_MALIGNO).next_to(gg, UP, buff=0.08)
        v_in = np.array([1.2, -0.3, 0]) * 0.8
        v_me = (pi.get_center() - o) * 0.9 * 0.5
        v_co = (gg.get_center() - o) * 0.9 * 0.5
        s1 = Arrow(o, o + v_in, buff=0, color=COR_SUAVE, stroke_width=5)
        s2 = Arrow(o + v_in, o + v_in + v_me, buff=0, color=COR_BENIGNO, stroke_width=5)
        s3 = Arrow(o + v_in + v_me, o + v_in + v_me + v_co, buff=0, color=COR_MALIGNO, stroke_width=5)
        res = Arrow(o, o + v_in + v_me + v_co, buff=0, color=COR_DESTAQUE, stroke_width=7)
        esquema = texto("esquema ilustrativo (2 das 32 dimensões)", 15, COR_SUAVE).move_to([2.6, -3.4, 0])
        self.play(FadeIn(part), FadeIn(pi), FadeIn(gg), FadeIn(rot_pi), FadeIn(rot_g), FadeIn(esquema))
        self.play(GrowArrow(s1))
        self.play(GrowArrow(s2))
        self.play(GrowArrow(s3))
        self.play(GrowArrow(res))
        self.play(part.animate.move_to(res.get_end()))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def montar_plano(self):
        """Plano (h, w) com a região viável h·w ≤ 30 e um gráfico de convergência ao lado."""
        plano = Axes(x_range=[0, 31, 5], y_range=[0, 31, 5], x_length=5.4, y_length=5.4, tips=False,
                     axis_config={"color": COR_SUAVE, "include_numbers": True, "font_size": 20}) \
            .move_to(LEFT * 3.4 + DOWN * 0.45)
        curva = [plano.c2p(h, M / h) for h in np.linspace(1, M, 80)]
        regiao = Polygon(plano.c2p(1, 1), *curva, plano.c2p(M, 1), stroke_width=0).set_fill(COR_BOM, 0.15)
        borda = VMobject(stroke_color=COR_BOM, stroke_width=2).set_points_smoothly(curva)
        rot_x = texto("h (altura)", 18, COR_SUAVE).next_to(plano.x_axis, DOWN, buff=0.4)
        rot_y = texto("w (largura)", 18, COR_SUAVE).rotate(PI / 2).next_to(plano.y_axis, LEFT, buff=0.45)
        rot_reg = VGroup(texto("h·w ≤ 30 (válido)", 16, COR_BOM).move_to(plano.c2p(10, 8)))
        rot_reg.add(Arrow(rot_reg[0].get_bottom() + LEFT * 0.3, plano.c2p(3, 3), buff=0.05, color=COR_BOM,
                          stroke_width=2, tip_length=0.12))
        fora = texto("fora: o reparo reduz w", 15, COR_SUAVE).move_to(plano.c2p(20, 20))

        conv = Axes(x_range=[0, 500, 100], y_range=[0.7, 1.5, 0.2], x_length=5.2, y_length=3.0, tips=False,
                    axis_config={"color": COR_SUAVE, "include_numbers": True, "font_size": 18}) \
            .move_to(RIGHT * 3.6 + UP * 0.5)
        rot_cx = texto("avaliações de f", 16, COR_SUAVE).next_to(conv.x_axis, DOWN, buff=0.35)
        rot_cy = texto("melhor f", 16, COR_SUAVE).next_to(conv, UP, buff=0.1).align_to(conv, LEFT)
        legenda = VGroup(
            texto("cor da partícula = f da sua solução:", 15, COR_SUAVE),
            VGroup(Dot(color=COR_F_BAIXO), texto("baixo (bom)", 15, COR_F_BAIXO),
                   Dot(color=COR_F_ALTO), texto("alto (ruim)", 15, "#9B7FD0")).arrange(RIGHT, buff=0.12),
            texto("+ 30 dimensões b_k que não cabem no desenho", 15, COR_SUAVE),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).to_corner(DR, buff=0.3)
        return VGroup(plano, regiao, borda, rot_x, rot_y, rot_reg, fora), VGroup(conv, rot_cx, rot_cy), legenda, plano, conv

    def animar_enxame(self, nome, tamanho_ponto=0.07):
        """Mostra as fotos reais da população do otimizador `nome` e a curva de convergência."""
        dados = OPT["otimizadores"][nome]
        grupo_plano, grupo_conv, legenda, plano, conv = self.montar_plano()
        self.play(FadeIn(grupo_plano), FadeIn(grupo_conv), FadeIn(legenda))

        snaps = dados["snapshots"]
        hist = dados["historico"]

        def pontos_de(s):
            return [plano.c2p(*hw) for hw in s["hw"]]

        particulas = VGroup(*[Dot(p, radius=tamanho_ponto, color=cor_f(f))
                              for p, f in zip(pontos_de(snaps[0]), snaps[0]["f"])])
        it = texto("iteração 0", 20, COR_TEXTO).next_to(plano, UP, buff=0.15)
        melhor = texto(f"melhor f = {br(min(hist[:AVALIACAO_DO_SNAPSHOT[nome](0)]), 3)}", 20, COR_DESTAQUE) \
            .next_to(it, RIGHT, buff=0.6)
        self.play(LaggedStart(*[FadeIn(p, scale=0.3) for p in particulas], lag_ratio=0.02), FadeIn(it), FadeIn(melhor))

        def trecho_curva(ate):
            ate = min(ate, len(hist))
            pts = [conv.c2p(i + 1, np.clip(hist[i], 0.7, 1.5)) for i in range(ate)]
            return VMobject(stroke_color=COR_DESTAQUE, stroke_width=3).set_points_as_corners(pts)

        curva = trecho_curva(AVALIACAO_DO_SNAPSHOT[nome](0))
        self.play(Create(curva))
        self.next_slide()

        for k in range(1, len(snaps)):
            ate = AVALIACAO_DO_SNAPSHOT[nome](k)
            novo_it = texto(f"iteração {k}", 20, COR_TEXTO).move_to(it)
            novo_melhor = texto(f"melhor f = {br(min(hist[:min(ate, len(hist))]), 3)}", 20, COR_DESTAQUE) \
                .move_to(melhor, aligned_edge=LEFT)
            self.play(*[p.animate.move_to(q).set_color(cor_f(f))
                        for p, q, f in zip(particulas, pontos_de(snaps[k]), snaps[k]["f"])],
                      Transform(curva, trecho_curva(ate)), Transform(it, novo_it), Transform(melhor, novo_melhor),
                      run_time=0.9)

        h, w = dados["forma"]
        final = VGroup(
            texto(f"Resultado: formato {h}×{w}, {h * w} features", 22, COR_TEXTO),
            texto(f"f = {br(dados['f'], 3)}", 22, COR_DESTAQUE),
        )
        if min(hist) >= hist[AVALIACAO_DO_SNAPSHOT[nome](0) - 1]:
            final.add(texto("nenhuma melhora em 500 avaliações: a melhor fonte\né uma das sorteadas no início",
                            17, COR_RUIM, line_spacing=1.1))
        final.arrange(DOWN, buff=0.1, aligned_edge=LEFT).next_to(conv, DOWN, buff=0.6).align_to(conv, LEFT)
        self.play(FadeOut(legenda), FadeIn(final))  # a legenda sai para dar lugar ao resultado
        self.next_slide()
        self.limpar_tudo()

    def slide_8_pso_enxame(self):
        self.trocar_cabecalho("PSO real: 50 partículas, 500 avaliações")
        self.animar_enxame("pso")

    # =================================================================
    def slide_9_epso(self):
        self.trocar_cabecalho("EPSO · Evolutionary PSO: os pesos também evoluem")
        passos = VGroup(
            VGroup(texto("1. Réplica e mutação dos pesos", 22, COR_DESTAQUE),
                   formula(r"w^{*} = w + \tau\, \mathcal{N}(0,1), \qquad \tau = 0{,}8", tamanho=34)),
            VGroup(texto("2. Original e réplica se movem (g perturbado; cooperação com prob. 0,9)", 22, COR_DESTAQUE),
                   formula(r"g^{*} = g + w^{G}\,\mathcal{N}(0,1) \qquad v^{t+1} = w^{I} v^{t} + w^{M}(p_i - x^{t})"
                           r" + w^{C}(g^{*} - x^{t})\,P", tamanho=32)),
            VGroup(texto("3. Seleção: sobrevive a de menor f, com os seus pesos", 22, COR_DESTAQUE),
                   texto("pesos bons se propagam ao longo das gerações · 2 avaliações por partícula por iteração",
                         18, COR_SUAVE)),
        )
        for p in passos:
            p.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        passos.arrange(DOWN, buff=0.45, aligned_edge=LEFT).move_to(DOWN * 0.2)
        for p in passos:
            self.play(FadeIn(p, shift=RIGHT * 0.2))
        self.next_slide()
        self.limpar()
        self.titulo = None
        self.trocar_cabecalho("EPSO real: mesmo orçamento, menos iterações")
        self.animar_enxame("epso")

    # =================================================================
    def slide_10_abc(self):
        self.trocar_cabecalho("ABC · Artificial Bee Colony: 25 fontes de alimento")
        passos = VGroup(
            VGroup(texto("1. Abelhas empregadas: mexem em UMA dimensão d, na direção de outra fonte k", 21, COR_DESTAQUE),
                   formula(r"v_{i,d} = x_{i,d} + \phi\,(x_{i,d} - x_{k,d}), \qquad \phi \sim U(-1, 1)", tamanho=34),
                   texto("se f(v) < f(x_i), a fonte é trocada; senão, conta uma falha", 17, COR_SUAVE)),
            VGroup(texto("2. Abelhas observadoras: escolhem as fontes boas com mais frequência", 21, COR_DESTAQUE),
                   formula(r"\text{fit}_i = \frac{1}{1 + f(x_i)}, \qquad p_i = \frac{\text{fit}_i}{\sum_k \text{fit}_k}",
                           tamanho=34)),
            VGroup(texto("3. Abelha exploradora: fonte com falhas demais é trocada por uma aleatória", 21, COR_DESTAQUE),
                   texto("limite de falhas = (32 dimensões + 50) / 2 = 41", 17, COR_SUAVE)),
        )
        for p in passos:
            p.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        passos.arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to(DOWN * 0.25)
        for p in passos:
            self.play(FadeIn(p, shift=RIGHT * 0.2))
        self.next_slide()
        self.limpar()
        self.titulo = None
        self.trocar_cabecalho("ABC real: as fontes andam uma dimensão por vez")
        self.animar_enxame("abc", tamanho_ponto=0.09)

    # =================================================================
    def slide_11_comparacao(self):
        self.trocar_cabecalho("Os três otimizadores, lado a lado")
        z_b = OPT["pacientes_ordenados"]["benigno"]
        z_m = OPT["pacientes_ordenados"]["maligno"]

        colunas = VGroup()
        for nome, rotulo in [("pso", "PSO"), ("epso", "EPSO"), ("abc", "ABC")]:
            d = OPT["otimizadores"][nome]
            cv = OPT["cv"][nome]
            h, w = d["forma"]
            sel = [k for k in range(M) if d["mascara"][k]]
            lado = 0.42 if h * w <= 4 else 0.2
            img_b = grade([z_b[k] for k in sel], h, w, lado)
            img_m = grade([z_m[k] for k in sel], h, w, lado)
            img_b.add(SurroundingRectangle(img_b, color=COR_BENIGNO, buff=0.03))
            img_m.add(SurroundingRectangle(img_m, color=COR_MALIGNO, buff=0.03))
            imgs = VGroup(img_b, img_m).arrange(RIGHT, buff=0.35)
            nomes_feat = ", ".join(f"F{ORDEM[k] + 1}" for k in sel) if len(sel) <= 3 else f"{len(sel)} features"
            col = VGroup(
                texto(rotulo, 32, COR_TEXTO, weight=BOLD),
                texto(f"formato {h}×{w} · {nomes_feat}", 18, COR_SUAVE),
                imgs,
                formula(r"f =", tex(d["f"], 3), tamanho=34),
                texto(f"acurácia bal. (CV): {br(cv['cv'], 3)}", 20, COR_DESTAQUE),
            ).arrange(DOWN, buff=0.25)
            colunas.add(col)
        colunas.arrange(RIGHT, buff=1.3, aligned_edge=UP).next_to(self.titulo, DOWN, buff=0.4)
        if colunas.height > 6.3:
            colunas.scale_to_fit_height(6.3).next_to(self.titulo, DOWN, buff=0.3)
        legenda = VGroup(Square(0.2).set_stroke(COR_BENIGNO, 3), texto("benigno", 16, COR_BENIGNO),
                         Square(0.2).set_stroke(COR_MALIGNO, 3), texto("maligno", 16, COR_MALIGNO)) \
            .arrange(RIGHT, buff=0.12).to_corner(DR, buff=0.35)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in colunas], lag_ratio=0.3), FadeIn(legenda))
        self.next_slide()

        # Destaque: menor f NÃO é melhor classificação
        alerta = texto("PSO e EPSO otimizam MELHOR (f menor)… e classificam PIOR.", 24, COR_RUIM) \
            .to_corner(DL, buff=0.35)
        self.play(Create(SurroundingRectangle(colunas[2], color=COR_DESTAQUE, buff=0.15)), FadeIn(alerta))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_12_o_que_a_eq1_premia(self):
        self.trocar_cabecalho("O que a Eq. 1 premia?")
        f = formula(r"f = \frac{1}{SM}\sum_{j \in S} D_j", r"\qquad", r"D_j = \frac{1}{N}\sum_{i}\left(z_{i,j} - z_{i+1,j}\right)^2",
                    tamanho=36).next_to(self.titulo, DOWN, buff=0.3)
        expl = texto("f é só a MÉDIA de uma contribuição D_j por feature escolhida", 20, COR_DESTAQUE) \
            .next_to(f, DOWN, buff=0.2)
        self.play(Write(f), FadeIn(expl))

        # Barras de D_j, na ordem do ranking
        D = np.array(OPT["D"])
        eixo = Axes(x_range=[0, 31, 5], y_range=[0, 2.2, 0.5], x_length=7.0, y_length=3.2, tips=False,
                    axis_config={"color": COR_SUAVE, "font_size": 16}, y_axis_config={"include_numbers": True}) \
            .move_to(LEFT * 2.6 + DOWN * 1.7)
        larg = eixo.x_axis.unit_size * 0.75
        barras = VGroup(*[Rectangle(width=larg, height=eixo.y_axis.unit_size * d, stroke_width=0)
                          .set_fill(COR_BENIGNO, 0.85).move_to(eixo.c2p(k + 1, 0), aligned_edge=DOWN)
                          for k, d in enumerate(D)])
        rot_x = texto("posição no ranking do Map", 16, COR_SUAVE).next_to(eixo, DOWN, buff=0.15)
        rot_y = texto("D_j", 18, COR_SUAVE).next_to(eixo.y_axis, UP, buff=0.1)
        self.play(Create(eixo), FadeIn(rot_x), FadeIn(rot_y),
                  LaggedStart(*[GrowFromEdge(b, DOWN) for b in barras], lag_ratio=0.03))
        nota_d = texto("(features importantes variam pouco dentro de cada classe → D_j menor)", 16, COR_SUAVE) \
            .next_to(rot_x, DOWN, buff=0.08)
        self.play(FadeIn(nota_d))
        self.next_slide()

        # Curva: f quando se escolhem as k features de menor D_j
        fk = np.array(OPT["f_por_k_menores"])          # k = 2..30
        g = Axes(x_range=[0, 31, 5], y_range=[0.7, 1.6, 0.2], x_length=4.6, y_length=3.2, tips=False,
                 axis_config={"color": COR_SUAVE, "font_size": 16, "include_numbers": True}) \
            .move_to(RIGHT * 4.2 + DOWN * 1.7)
        pts = [g.c2p(k, v) for k, v in zip(range(2, M + 1), fk)]
        linha = VMobject(stroke_color=COR_DESTAQUE, stroke_width=4).set_points_as_corners(pts)
        rot_g = texto("menor f possível usando k features", 16, COR_SUAVE).next_to(g, UP, buff=0.1)
        rot_gx = texto("k (número de pixels)", 16, COR_SUAVE).next_to(g, DOWN, buff=0.15)
        minimo = Dot(pts[0], color=COR_RUIM, radius=0.1)
        rot_min = texto("mínimo: k = 2", 18, COR_RUIM).next_to(minimo, UP, buff=0.35).shift(RIGHT * 0.7)
        self.play(Create(g), FadeIn(rot_g), FadeIn(rot_gx))
        self.play(Create(linha), run_time=2)
        self.play(FadeIn(minimo), FadeIn(rot_min))
        self.next_slide()

        conclusao = VGroup(
            texto("A média nunca é menor que a média das 2 menores  →  ótimo global = imagem 1×2", 20, COR_RUIM),
            texto("E f não muda se os pixels forem embaralhados: a Eq. 1 ignora a posição de cada pixel.", 20, COR_RUIM),
        ).arrange(DOWN, buff=0.12).move_to(VGroup(f, expl))
        self.play(FadeOut(VGroup(f, expl)), FadeIn(conclusao))
        self.next_slide()

        # Prova visual da segunda frase: embaralhar pixels de um par não muda a diferença média
        self.limpar_tudo()
        self.trocar_cabecalho("f ignora o arranjo dos pixels")
        seq = OPT["sequencia"]
        za, zb = np.array(seq["z"][0][:16]), np.array(seq["z"][1][:16])
        ga, gb = grade(za, 4, 4, 0.55), grade(zb, 4, 4, 0.55)
        valor = formula(r"\text{diferença média} =", tex(seq["mse"][0], 3), tamanho=40)
        valor[1].set_color(COR_DESTAQUE)
        VGroup(VGroup(ga, gb).arrange(RIGHT, buff=1.0), valor).arrange(DOWN, buff=0.6)
        self.play(FadeIn(ga), FadeIn(gb), FadeIn(valor))
        perm = np.random.default_rng(1).permutation(16)
        pos_a = [ga[k].get_center() for k in range(16)]
        pos_b = [gb[k].get_center() for k in range(16)]
        self.play(*[ga[k].animate.move_to(pos_a[perm[k]]) for k in range(16)],
                  *[gb[k].animate.move_to(pos_b[perm[k]]) for k in range(16)], run_time=2, path_arc=PI / 3)
        self.play(Indicate(valor[1], color=COR_DESTAQUE))
        msg = texto("Mesmo embaralhando os pixels (igual nos dois), f fica igual.\n"
                    "Quem cria a estrutura espacial é só o Map, não o Optimize.", 22, COR_TEXTO, line_spacing=1.2) \
            .to_edge(DOWN, buff=0.5)
        self.play(FadeIn(msg))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_13_saida(self):
        self.trocar_cabecalho("Saída do OPTIMIZE")
        itens = VGroup(
            texto("1.  O formato da imagem: h × w", 26),
            texto("2.  Quais features entram (a máscara b)", 26),
            texto("3.  As imagens de todos os pacientes: N × h × w", 26),
            texto("Próxima etapa → LEARN: a CNN aprende com essas imagens", 26, COR_DESTAQUE),
        ).arrange(DOWN, buff=0.5, aligned_edge=LEFT).move_to(DOWN * 0.2)
        itens[3].shift(DOWN * 0.3)
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.3) for i in itens], lag_ratio=0.4), run_time=2.5)
        self.next_slide()
