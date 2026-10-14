"""
CENA 6 — Análise crítica: por que o MOL não funcionou, e o que fica de conclusão
================================================================================

Roteiro (cada método slide_... termina num `self.next_slide()`, onde a apresentação pausa):
   1. Título
   2. Roteiro da análise
   3. Defeitos 1 e 2: a Eq. 1 ignora o formato e a posição dos pixels
   4. Defeito 3: a Eq. 1 premia poucas features (f menor veio com classificação pior)
   5. Teste de controle 1: otimizar é melhor que sortear?
   6. Teste de controle 2: o formato 2D e a ordem do Map importam?
   7. Teste de controle 3: transformar em imagem ajuda?
   8. Síntese dos três testes
   9. MOL v2: a versão corrigida (o que muda e como fica o arranjo)
  10. MOL v2: o resultado
  11. Limitações
  12. Conclusão e recomendação
  13. Encerramento

COMO RODAR (de dentro da pasta apresentacao/):
    .venv/bin/manim-slides render -ql cena6_critica.py Cena6Critica   # prévia rápida
    .venv/bin/manim-slides render cena6_critica.py Cena6Critica       # 1080p
    .venv/bin/manim-slides present Cena6Critica                       # apresentar

Os números vêm de results/controles/ e results/mol_v2/ (carregados em dados/mol_dados.json, chave "critica").
"""
import numpy as np
from manim import *

from estilo import (COR_BENIGNO, COR_BOM, COR_DESTAQUE, COR_FUNDO, COR_MALIGNO, COR_RUIM,
                    COR_SUAVE, COR_TEXTO, DADOS, CenaBase, br, formula, pixel, tex, texto)

C = DADOS["critica"]
OPT = DADOS["optimize"]
NOMES = DADOS["feature_names"]
COR_ABC, COR_PSO, COR_EPSO = COR_BOM, COR_MALIGNO, COR_RUIM
COR_OTIM = {"pso": COR_PSO, "epso": COR_EPSO, "abc": COR_ABC}

# Tipo de medida de cada feature (para colorir o mapa do arranjo do MOL v2)
TIPOS = {"radius": "tamanho", "perimeter": "tamanho", "area": "tamanho", "texture": "textura",
         "smoothness": "suavidade", "compactness": "forma", "concavity": "forma", "concave points": "forma",
         "symmetry": "simetria", "fractal_dimension": "dim. fractal"}
COR_TIPO = {"tamanho": "#F58518", "textura": "#54A24B", "suavidade": "#4C78A8", "forma": "#B279A2",
            "simetria": "#E4B400", "dim. fractal": "#8A8F98"}


def barras_h(itens, v_min=0.90, v_max=1.0, larg_max=5.5, alt=0.5, tam=19, larg_rotulo=4.3):
    """
    Barras horizontais. itens = lista de (rótulo, valor, cor).
    Devolve um VGroup onde cada linha é VGroup(rótulo, barra, valor).
    """
    linhas = VGroup()
    for i, (rot, v, cor) in enumerate(itens):
        barra = Rectangle(width=max((v - v_min) / (v_max - v_min) * larg_max, 0.03), height=alt * 0.72,
                          stroke_width=0).set_fill(cor, 0.95)
        barra.move_to([0, -i * alt, 0], aligned_edge=LEFT)
        t = texto(rot, tam, COR_TEXTO)
        if t.width > larg_rotulo:
            t.scale_to_fit_width(larg_rotulo)
        t.move_to([-0.2 - t.width / 2, -i * alt, 0])
        val = texto(br(v, 3), tam, cor, weight=BOLD).next_to(barra, RIGHT, buff=0.12)
        linhas.add(VGroup(t, barra, val))
    return linhas


class Cena6Critica(CenaBase):
    def construct(self):
        self.slide_1_titulo()
        self.slide_2_roteiro()
        self.slide_3_formato_e_posicao()
        self.slide_4_poucas_features()
        self.slide_5_teste_1()
        self.slide_6_teste_2()
        self.slide_7_teste_3()
        self.slide_8_sintese()
        self.slide_9_mol_v2()
        self.slide_10_resultado_v2()
        self.slide_11_limitacoes()
        self.slide_12_conclusao()
        self.slide_13_encerramento()

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
        etapa = texto("ANÁLISE CRÍTICA", 30, COR_DESTAQUE)
        titulo = texto("Por quê?", 96, weight=BOLD)
        sub = texto("O que cada parte do MOL realmente faz", 32, COR_SUAVE)
        VGroup(etapa, titulo, sub).arrange(DOWN, buff=0.35)
        self.play(FadeIn(etapa, shift=DOWN * 0.3))
        self.play(Write(titulo))
        self.play(FadeIn(sub, shift=UP * 0.3))
        self.next_slide()
        self.limpar()

    # =================================================================
    def slide_2_roteiro(self):
        self.trocar_cabecalho("Quatro passos para responder")

        def passo(n, nome, desc, cor):
            r = RoundedRectangle(width=2.9, height=2.3, corner_radius=0.15, stroke_color=cor, stroke_width=3)
            c = VGroup(texto(str(n), 40, cor, weight=BOLD), texto(nome, 21, cor, weight=BOLD),
                       texto(desc, 16, COR_TEXTO, line_spacing=1.0)).arrange(DOWN, buff=0.15).move_to(r)
            return VGroup(r, c)

        passos = VGroup(
            passo(1, "Defeitos de projeto", "o que a Eq. 1 mede\n(matemática)", COR_RUIM),
            passo(2, "Testes de controle", "cada parte ajuda?\n(experimentos)", COR_DESTAQUE),
            passo(3, "Versão corrigida", "MOL v2: consertando\nos defeitos", COR_BOM),
            passo(4, "Conclusão", "o que recomendar\npara o BI-RADS", COR_BENIGNO),
        ).arrange(RIGHT, buff=0.4).move_to(DOWN * 0.2)
        setas = VGroup(*[Arrow(passos[i].get_right(), passos[i + 1].get_left(), buff=0.04, color=COR_SUAVE,
                               stroke_width=3, max_tip_length_to_length_ratio=0.4) for i in range(3)])
        self.play(LaggedStart(*[FadeIn(p, shift=UP * 0.2) for p in passos], lag_ratio=0.3), FadeIn(setas))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_3_formato_e_posicao(self):
        self.trocar_cabecalho("Defeitos 1 e 2: a Eq. 1 ignora o formato e a posição dos pixels")
        f = formula(r"f = \frac{1}{N}\sum_{i=1}^{N-1}\frac{1}{SM}\sum_{j=1}^{SM}\left(X_{i,j} - X_{i+1,j}\right)^2",
                    tamanho=34).next_to(self.titulo, DOWN, buff=0.3)
        self.play(Write(f))

        z = OPT["pacientes_ordenados"]["exemplo"][:16]
        lado = 0.28
        pixels = VGroup(*[pixel(v, lado) for v in z])

        def posicoes(h, w, centro=LEFT * 2.8 + DOWN * 0.9):
            g = VGroup(*[Square(lado) for _ in range(16)]).arrange_in_grid(rows=h, cols=w, buff=0).move_to(centro)
            return [q.get_center() for q in g]

        for p, pos in zip(pixels, posicoes(4, 4)):
            p.move_to(pos)
        rot = texto("as mesmas 16 features, formato 4×4", 24, COR_TEXTO).move_to(RIGHT * 3.6 + DOWN * 2.2)
        valor = VGroup(formula("f =", tex(OPT["f_formas"]["4x4"], 3), tamanho=56),
                       texto("(valor real da Eq. 1 para estas 16 features)", 16, COR_SUAVE)) \
            .arrange(DOWN, buff=0.2).move_to(RIGHT * 3.6 + DOWN * 0.5)
        valor[0][1].set_color(COR_DESTAQUE)
        self.play(FadeIn(pixels), FadeIn(rot), FadeIn(valor))
        self.next_slide()

        def mudar(h, w, nome, perm=None):
            pos = posicoes(h, w)
            if perm is not None:
                pos = [pos[k] for k in perm]
            novo_rot = texto(nome, 24, COR_TEXTO).move_to(rot)
            self.play(*[p.animate.move_to(q) for p, q in zip(pixels, pos)], Transform(rot, novo_rot), run_time=1.3)
            self.play(Indicate(valor[0][1], color=COR_BOM, scale_factor=1.15), run_time=0.7)

        mudar(2, 8, "formato 2×8")
        mudar(8, 2, "formato 8×2")
        mudar(16, 1, "formato 16×1")
        self.next_slide()
        perm = np.random.default_rng(3).permutation(16)
        mudar(4, 4, "4×4 com os pixels embaralhados", perm)
        self.next_slide()

        conclusao = VGroup(
            texto("f é sempre o mesmo: a fórmula compara pixel j com pixel j, onde quer que ele esteja.", 20, COR_TEXTO),
            texto("Então o otimizador não tem motivo para preferir nenhum formato nem arranjo: h e w saem por acaso.",
                  20, COR_RUIM),
        ).arrange(DOWN, buff=0.12).to_edge(DOWN, buff=0.2)
        self.play(FadeOut(rot), FadeIn(conclusao))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_4_poucas_features(self):
        self.trocar_cabecalho("Defeito 3: a Eq. 1 premia imagens com poucas features")
        t1 = C["teste1"]

        # Esquerda: menor f possível com k features (mínimo em k = 2)
        fk = OPT["f_por_k_menores"]
        g = Axes(x_range=[0, 31, 5], y_range=[0.7, 1.6, 0.2], x_length=5.0, y_length=3.4, tips=False,
                 axis_config={"color": COR_SUAVE, "font_size": 18, "include_numbers": True}).move_to(LEFT * 3.7 + DOWN * 0.3)
        pts = [g.c2p(k, v) for k, v in zip(range(2, 31), fk)]
        linha = VMobject(stroke_color=COR_DESTAQUE, stroke_width=4).set_points_as_corners(pts)
        rot_g = texto("menor f possível usando k features", 17, COR_SUAVE).next_to(g, UP, buff=0.1)
        rot_gx = texto("k (número de features na imagem)", 16, COR_SUAVE).next_to(g, DOWN, buff=0.25)
        minimo = Dot(pts[0], color=COR_RUIM, radius=0.1)
        rot_min = texto("← mínimo: k = 2", 18, COR_RUIM).move_to(g.c2p(11.5, 0.775))
        self.play(Create(g), FadeIn(rot_g), FadeIn(rot_gx))
        self.play(Create(linha), run_time=1.5)
        self.play(FadeIn(minimo), FadeIn(rot_min))
        expl = texto("f é uma MÉDIA por feature:\ntirar features nunca aumenta o mínimo.", 17, COR_SUAVE, line_spacing=1.1) \
            .next_to(rot_gx, DOWN, buff=0.2)
        self.play(FadeIn(expl))
        self.next_slide()

        # Direita: f contra acurácia, para as 18 soluções testadas
        d = Axes(x_range=[0.7, 1.6, 0.2], y_range=[0.90, 0.98, 0.02], x_length=5.0, y_length=3.4, tips=False,
                 axis_config={"color": COR_SUAVE, "font_size": 18, "include_numbers": True},
                 y_axis_config={"decimal_number_config": {"num_decimal_places": 2}}).move_to(RIGHT * 3.6 + DOWN * 0.3)
        rot_d = texto("f (Eq. 1) contra acurácia balanceada (CV)", 17, COR_SUAVE).next_to(d, UP, buff=0.1)
        rot_dx = texto("f da solução  (← o otimizador empurra para cá)", 16, COR_SUAVE).next_to(d, DOWN, buff=0.25)
        pontos = VGroup(*[Dot(d.c2p(a["f"], a["cv"]), radius=0.07, color=COR_SUAVE) for a in t1["aleatorias"]])
        otim = VGroup()
        for nome, o in t1["otimizadas"].items():
            p = Dot(d.c2p(o["f"], o["cv"]), radius=0.11, color=COR_OTIM[nome])
            otim.add(VGroup(p, texto(nome.upper(), 16, COR_OTIM[nome]).next_to(p, RIGHT, buff=0.1)))
        self.play(Create(d), FadeIn(rot_d), FadeIn(rot_dx))
        self.play(LaggedStart(*[FadeIn(p, scale=0.4) for p in pontos], lag_ratio=0.05))
        self.play(LaggedStart(*[FadeIn(o) for o in otim], lag_ratio=0.3))
        leg = texto("cinza: 15 soluções sorteadas", 15, COR_SUAVE).move_to(d.c2p(1.3, 0.908))
        self.play(FadeIn(leg))
        self.next_slide()

        self.play(FadeOut(expl))
        conclusao = VGroup(
            texto(f"Nas 18 soluções testadas, f MENOR veio junto com classificação PIOR (correlação {br(t1['correlacao_f_cv'], 2)}).",
                  20, COR_RUIM),
            texto("O que explica isso é o número de features: quanto melhor a otimização, menos informação sobra.", 19, COR_SUAVE),
        ).arrange(DOWN, buff=0.12).to_edge(DOWN, buff=0.2)
        self.play(FadeIn(conclusao))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_5_teste_1(self):
        self.trocar_cabecalho("Teste de controle 1: otimizar é melhor que sortear?")
        t1 = C["teste1"]
        expl = texto("15 soluções com formato e features sorteados, avaliadas com o mesmo protocolo (10 folds).",
                     19, COR_SUAVE).next_to(self.titulo, DOWN, buff=0.25)
        self.play(FadeIn(expl))

        ax = Axes(x_range=[0, 32, 5], y_range=[0.90, 0.985, 0.02], x_length=6.4, y_length=3.8, tips=False,
                  axis_config={"color": COR_SUAVE, "font_size": 18, "include_numbers": True},
                  y_axis_config={"decimal_number_config": {"num_decimal_places": 2}}).move_to(LEFT * 3.3 + DOWN * 0.45)
        rot_x = texto("número de features na imagem", 16, COR_SUAVE).next_to(ax, DOWN, buff=0.25)
        rot_y = texto("acurácia balanceada (CV)", 16, COR_SUAVE).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
        self.play(Create(ax), FadeIn(rot_x), FadeIn(rot_y))

        pontos = VGroup(*[Dot(ax.c2p(a["n"], a["cv"]), radius=0.085, color=COR_TEXTO) for a in t1["aleatorias"]])
        self.play(LaggedStart(*[FadeIn(p, scale=0.4) for p in pontos], lag_ratio=0.06))
        media = DashedLine(ax.c2p(0, t1["media_aleatorias"]), ax.c2p(32, t1["media_aleatorias"]), color=COR_TEXTO,
                           stroke_width=2)
        self.play(Create(media))
        self.next_slide()

        linhas, legenda = VGroup(), VGroup(
            VGroup(Dot(color=COR_TEXTO, radius=0.08), texto(f"sorteadas (média {br(t1['media_aleatorias'], 3)})", 17)).arrange(RIGHT, buff=0.12))
        for nome in ["abc", "pso", "epso"]:
            o = t1["otimizadas"][nome]
            linhas.add(Line(ax.c2p(0, o["cv"]), ax.c2p(32, o["cv"]), color=COR_OTIM[nome], stroke_width=4))
            p = t1["p"][nome]
            legenda.add(VGroup(Line(ORIGIN, RIGHT * 0.4, color=COR_OTIM[nome], stroke_width=4),
                               texto(f"{nome.upper()} otimizado: {br(o['cv'], 3)}  (p = {br(p, 3)})", 17, COR_OTIM[nome]))
                        .arrange(RIGHT, buff=0.12))
        legenda.arrange(DOWN, buff=0.22, aligned_edge=LEFT).move_to(RIGHT * 3.6 + UP * 0.8)
        self.play(LaggedStart(*[Create(l) for l in linhas], lag_ratio=0.3), FadeIn(legenda))
        n_melhores = sum(1 for a in t1["aleatorias"] if a["cv"] > t1["otimizadas"]["abc"]["cv"])
        nota = VGroup(
            texto("As 15 sorteadas superam o PSO e o EPSO.", 18, COR_TEXTO),
            texto(f"{n_melhores} das 15 superam o ABC.", 18, COR_TEXTO),
            texto("Com o mesmo nº de features (16–17),\nsorteio e ABC empatam.", 17, COR_SUAVE, line_spacing=1.1),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT).next_to(legenda, DOWN, buff=0.5).align_to(legenda, LEFT)
        self.play(FadeIn(nota))
        self.next_slide()
        resposta = texto("Resposta: não. Otimizar a Eq. 1 dá um resultado pior do que sortear.", 23, COR_RUIM) \
            .to_edge(DOWN, buff=0.2)
        self.play(FadeIn(resposta))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_6_teste_2(self):
        self.trocar_cabecalho("Teste de controle 2: o formato 2D e a ordem do Map importam?")
        t2 = C["teste2"]
        expl = texto("As MESMAS 16 features (as primeiras do ranking). Só muda o formato ou a ordem dos pixels.",
                     19, COR_SUAVE).next_to(self.titulo, DOWN, buff=0.25)
        self.play(FadeIn(expl))

        z = OPT["pacientes_ordenados"]["exemplo"][:16]
        perm = np.random.default_rng(3).permutation(16)
        formas = [(4, 4, None), (2, 8, None), (16, 1, None), (1, 16, None), (4, 4, perm)]
        itens = [(f"{nome}", v, COR_SUAVE if "embaralhado" in nome else COR_BENIGNO) for nome, v in t2["barras"]]
        barras = barras_h(itens, alt=0.7, larg_rotulo=2.6).move_to(RIGHT * 0.3 + DOWN * 0.25)
        minis = VGroup()
        for (h, w, p_), linha in zip(formas, barras):
            vals = [z[k] for k in (p_ if p_ is not None else range(16))]
            lado = 0.16 if max(h, w) <= 8 else 0.09
            g = VGroup(*[pixel(v, lado) for v in vals]).arrange_in_grid(rows=h, cols=w, buff=0)
            if h == 16:
                g.rotate(0)  # 16x1 fica de pé; cabe na linha porque o pixel é pequeno
                g.scale_to_fit_height(0.7)
            g.next_to(linha[0], LEFT, buff=0.3)
            minis.add(g)
        self.play(LaggedStart(*[FadeIn(VGroup(l[0], m)) for l, m in zip(barras, minis)], lag_ratio=0.15))
        self.play(LaggedStart(*[GrowFromEdge(l[1], LEFT) for l in barras], lag_ratio=0.15),
                  FadeIn(VGroup(*[l[2] for l in barras])))
        eixo = texto("acurácia balanceada (CV de 10 folds; o eixo começa em 0,90)", 15, COR_SUAVE) \
            .next_to(barras, DOWN, buff=0.25)
        self.play(FadeIn(eixo))
        self.next_slide()

        ps = VGroup(
            texto(f"4×4 contra 16×1 (vetor):  p = {br(t2['p_formato'], 2)}", 20, COR_TEXTO),
            texto(f"4×4 contra 4×4 embaralhado:  p = {br(t2['p_ordem'], 2)}", 20, COR_TEXTO),
        ).arrange(RIGHT, buff=1.0).next_to(eixo, DOWN, buff=0.2).set_x(0)
        resposta = texto("Resposta: não. Nenhuma diferença é significativa; embaralhar os pixels não piora.", 23, COR_RUIM) \
            .to_edge(DOWN, buff=0.2)
        self.play(FadeIn(ps))
        self.play(FadeIn(resposta))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_7_teste_3(self):
        self.trocar_cabecalho("Teste de controle 3: transformar em imagem ajuda?")
        t3 = C["teste3"]
        expl = texto("As mesmas features numa rede neural comum (vetor, sem convolução), com o mesmo treino.",
                     19, COR_SUAVE).next_to(self.titulo, DOWN, buff=0.25)
        self.play(FadeIn(expl))

        itens, notas_p = [], []
        for nome, cnn, mlp, p in t3["pares"]:
            forma = "4×4" if "16" in nome else "5×6"
            itens.append((f"{nome} · CNN (imagem {forma})", cnn, COR_BENIGNO))
            itens.append((f"{nome} · rede comum (vetor)", mlp, COR_DESTAQUE))
            notas_p.append(p)
        itens.append(("30 features · regressão logística", t3["lr"], COR_MALIGNO))
        barras = barras_h(itens, alt=0.7, larg_rotulo=4.6, larg_max=4.6).move_to(DOWN * 0.3)
        barras.shift(RIGHT * (0.3 - barras.get_center()[0]))
        self.play(LaggedStart(*[FadeIn(l[0]) for l in barras], lag_ratio=0.1))
        self.play(LaggedStart(*[GrowFromEdge(l[1], LEFT) for l in barras], lag_ratio=0.15),
                  FadeIn(VGroup(*[l[2] for l in barras])))
        chaves = VGroup()
        for k, p in enumerate(notas_p):
            par = VGroup(barras[2 * k][2], barras[2 * k + 1][2])
            ch = Brace(par, RIGHT, color=COR_SUAVE)
            chaves.add(VGroup(ch, texto(f"p = {br(p, 2)}", 18, COR_SUAVE).next_to(ch, RIGHT, buff=0.1)))
        self.play(FadeIn(chaves))
        eixo = texto("acurácia balanceada (CV de 10 folds; o eixo começa em 0,90)", 15, COR_SUAVE) \
            .next_to(barras, DOWN, buff=0.3)
        self.play(FadeIn(eixo))
        self.next_slide()
        resposta = texto("Resposta: não. A convolução não acrescenta nada a uma rede comum com as mesmas features.",
                         22, COR_RUIM).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(resposta))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_8_sintese(self):
        self.trocar_cabecalho("Síntese: o que cada parte do MOL faz nesta base")
        linhas_txt = [
            ("Parte do MOL", "Pergunta", "Resposta", "Evidência"),
            ("Optimize (Eq. 1)", "Otimizar é melhor que sortear?", "Não: piora", "15 sorteadas > PSO e EPSO"),
            ("Formato e ordem (Map)", "O formato 2D e a ordem ajudam?", "Não", "4×4 = 16×1 = embaralhado"),
            ("Imagem + CNN (Learn)", "Virar imagem ajuda?", "Não", "CNN = rede comum = LR"),
        ]
        larguras = [3.3, 4.3, 1.9, 3.9]
        tab = VGroup()
        for i, linha in enumerate(linhas_txt):
            cels = VGroup()
            for j, (v, w) in enumerate(zip(linha, larguras)):
                caixa = Rectangle(width=w, height=0.75, stroke_color=COR_SUAVE, stroke_width=1)
                cor = COR_DESTAQUE if i == 0 else (COR_RUIM if j == 2 else COR_TEXTO)
                t = texto(v, 19, cor, weight=BOLD if i == 0 or j == 2 else NORMAL)
                if t.width > w - 0.15:
                    t.scale_to_fit_width(w - 0.15)
                cels.add(VGroup(caixa, t.move_to(caixa)))
            tab.add(cels.arrange(RIGHT, buff=0))
        tab.arrange(DOWN, buff=0).move_to(UP * 0.5)
        self.play(FadeIn(tab[0]))
        for l in tab[1:]:
            self.play(FadeIn(l, shift=DOWN * 0.1))
        self.next_slide()
        nota = VGroup(
            texto("O melhor resultado com rede neural vem de usar TODAS as 30 features (≈ 0,97),", 21, COR_TEXTO),
            texto("como imagem, como vetor ou com regressão logística. O MOL só perde quando descarta features.", 21, COR_TEXTO),
        ).arrange(DOWN, buff=0.12).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(nota))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def mapa_arranjo(self, colunas, lado=0.62):
        """Grade 5x6 com uma célula por feature, colorida pelo tipo de medida."""
        g = VGroup()
        for c in colunas:
            base = NOMES[c].rsplit("_", 1)[0]
            q = Square(lado).set_fill(COR_TIPO[TIPOS[base]], 0.9).set_stroke(COR_FUNDO, 2)
            g.add(q)
        return g.arrange_in_grid(rows=5, cols=6, buff=0)

    def slide_9_mol_v2(self):
        self.trocar_cabecalho("MOL v2: consertando os três defeitos")
        linhas_txt = [
            ("Defeito do MOL original", "Correção no MOL v2"),
            ("A Eq. 1 ignora a posição dos pixels", "O objetivo É a posição: features correlacionadas viram vizinhas"),
            ("O formato sai por acaso", "Formato fixo, o mais quadrado possível (30 features → 5×6)"),
            ("O mínimo é sempre 2 features", "O arranjo não descarta features (seleção é separada e opcional)"),
        ]
        larguras = [4.8, 8.2]
        tab = VGroup()
        for i, linha in enumerate(linhas_txt):
            cels = VGroup()
            for j, (v, w) in enumerate(zip(linha, larguras)):
                caixa = Rectangle(width=w, height=0.7, stroke_color=COR_SUAVE, stroke_width=1)
                cor = COR_DESTAQUE if i == 0 else (COR_RUIM if j == 0 else COR_BOM)
                t = texto(v, 19, cor, weight=BOLD if i == 0 else NORMAL)
                if t.width > w - 0.2:
                    t.scale_to_fit_width(w - 0.2)
                cels.add(VGroup(caixa, t.move_to(caixa)))
            tab.add(cels.arrange(RIGHT, buff=0))
        tab.arrange(DOWN, buff=0).move_to(UP * 0.6)
        self.play(FadeIn(tab[0]))
        for l in tab[1:]:
            self.play(FadeIn(l, shift=DOWN * 0.1))
        nota = texto("A ideia vem da literatura de tabela para imagem (DeepInsight, REFINED, IGTD).\n"
                     "A CNN e o protocolo são os mesmos, para a comparação ser justa.", 19, COR_SUAVE, line_spacing=1.1) \
            .to_edge(DOWN, buff=0.7)
        self.play(FadeIn(nota))
        self.next_slide()

        # O arranjo: aleatório -> otimizado (as células deslizam)
        self.play(FadeOut(VGroup(tab, nota)))
        self.trocar_cabecalho("MOL v2: features correlacionadas ficam em pixels vizinhos")
        arr = C["v2"]["arranjos"]
        m_al = self.mapa_arranjo(arr["aleatorio"]["colunas"]).move_to(LEFT * 3.6 + DOWN * 0.3)
        m_ot = self.mapa_arranjo(arr["correlacao"]["colunas"]).move_to(RIGHT * 3.6 + DOWN * 0.3)
        r_al = VGroup(texto("posição aleatória", 22, COR_SUAVE),
                      texto(f"|correlação| entre vizinhos = {br(arr['aleatorio']['correlacao_vizinhos'], 2)}", 18, COR_SUAVE)) \
            .arrange(DOWN, buff=0.08).next_to(m_al, UP, buff=0.2)
        r_ot = VGroup(texto("arranjo otimizado (MOL v2)", 22, COR_BOM),
                      texto(f"|correlação| entre vizinhos = {br(arr['correlacao']['correlacao_vizinhos'], 2)}", 18, COR_BOM)) \
            .arrange(DOWN, buff=0.08).next_to(m_ot, UP, buff=0.2)
        self.play(FadeIn(m_al), FadeIn(r_al))
        self.next_slide()

        # cada célula do mapa aleatório desliza para a posição que tem no mapa otimizado
        pos_ot = {c: m_ot[i].get_center() for i, c in enumerate(arr["correlacao"]["colunas"])}
        copia = m_al.copy()
        self.play(*[copia[i].animate.move_to(pos_ot[c]) for i, c in enumerate(arr["aleatorio"]["colunas"])],
                  FadeIn(r_ot), run_time=2.5)
        legenda = VGroup(*[VGroup(Square(0.22).set_fill(cor, 0.9).set_stroke(width=0), texto(tipo, 16, COR_TEXTO))
                           .arrange(RIGHT, buff=0.1) for tipo, cor in COR_TIPO.items()]) \
            .arrange(RIGHT, buff=0.45).to_edge(DOWN, buff=0.75)
        nota2 = texto("Cada célula é uma feature, colorida pelo tipo de medida. No arranjo otimizado, tamanho e forma ficam agrupados.",
                      17, COR_SUAVE).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(legenda), FadeIn(nota2))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_10_resultado_v2(self):
        self.trocar_cabecalho("MOL v2: o arranjo funciona, mas a classificação não muda")
        v = C["v2"]
        cores = {"v2": COR_BOM, "controle": COR_SUAVE, "vetor": COR_DESTAQUE, "baseline": COR_MALIGNO, "mol": COR_RUIM}
        itens = [(nome, val, cores[tipo]) for nome, val, tipo in v["barras"]]
        barras = barras_h(itens, alt=0.68, larg_rotulo=4.8, larg_max=4.6).move_to(UP * 0.55)
        barras.shift(RIGHT * (0.2 - barras.get_center()[0]))
        self.play(LaggedStart(*[FadeIn(l[0]) for l in barras], lag_ratio=0.1))
        self.play(LaggedStart(*[GrowFromEdge(l[1], LEFT) for l in barras], lag_ratio=0.15),
                  FadeIn(VGroup(*[l[2] for l in barras])))
        eixo = texto("acurácia balanceada (CV de 10 folds, média de 3 sementes; o eixo começa em 0,90)", 15, COR_SUAVE) \
            .next_to(barras, DOWN, buff=0.25)
        self.play(FadeIn(eixo))
        self.next_slide()

        ps = VGroup(
            texto(f"otimizado contra aleatório:  p = {br(v['p_aleatorio'], 2)}", 19, COR_TEXTO),
            texto(f"otimizado contra o pior arranjo:  p = {br(v['p_pior'], 2)}", 19, COR_TEXTO),
            texto(f"MOL v2 contra regressão logística:  p = {br(v['p_lr'], 2)}", 19, COR_TEXTO),
            texto(f"MOL v2 contra MOL original:  p = {br(v['p_mol'], 2)}", 19, COR_TEXTO),
        ).arrange_in_grid(rows=2, cols=2, buff=(1.0, 0.15), cell_alignment=LEFT).next_to(eixo, DOWN, buff=0.3)
        self.play(FadeIn(ps))
        conclusoes = VGroup(
            texto("O v2 elimina o prejuízo do MOL original (volta para ≈ 0,97).", 21, COR_BOM),
            texto("Mas otimizado, aleatório e pior arranjo empatam: a estrutura espacial não ajuda nesta base.", 21, COR_RUIM),
        ).arrange(DOWN, buff=0.1).to_edge(DOWN, buff=0.2)
        self.play(FadeIn(conclusoes))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_11_limitacoes(self):
        self.trocar_cabecalho("O que estes resultados NÃO permitem afirmar")
        itens = VGroup(
            texto("•  Uma base só, e fácil: quase todo modelo passa de 0,95.", 24),
            texto("    Diferenças menores que ≈ 0,01 não são detectáveis com 455 pacientes e 10 folds.", 19, COR_SUAVE),
            texto("•  Só 30 features. Métodos de imagem relatam ganhos com centenas ou milhares", 24),
            texto("    de features correlacionadas (por exemplo, expressão gênica). Isso não foi testado.", 19, COR_SUAVE),
            texto("•  A CNN é a pequena do artigo (1 convolução). Redes maiores não foram testadas.", 24),
            texto("•  Os defeitos da Eq. 1, esses sim, valem para qualquer base: são matemáticos.", 24, COR_DESTAQUE),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT).move_to(DOWN * 0.2)
        if itens.width > 13.2:
            itens.scale_to_fit_width(13.2)
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.2) for i in itens], lag_ratio=0.25), run_time=3)
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_12_conclusao(self):
        self.trocar_cabecalho("Conclusão")
        itens = VGroup(
            texto("1.  O MOL foi implementado fielmente e testado com o protocolo do artigo.", 24),
            texto("2.  A função objetivo (Eq. 1) tem três defeitos de projeto, demonstrados matematicamente.", 24),
            texto("3.  Nesta base, nenhuma parte do MOL melhora a classificação; otimizar a Eq. 1 piora.", 24),
            texto("4.  A versão corrigida (MOL v2) elimina o prejuízo, mas não supera modelos simples.", 24),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to(UP * 0.7)
        if itens.width > 13.2:
            itens.scale_to_fit_width(13.2)
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.2) for i in itens], lag_ratio=0.3), run_time=3)
        self.next_slide()

        caixa = RoundedRectangle(width=12.6, height=1.9, corner_radius=0.15, stroke_color=COR_DESTAQUE, stroke_width=3) \
            .move_to(DOWN * 2.3)
        rec = VGroup(
            texto("Recomendação para o BI-RADS", 24, COR_DESTAQUE, weight=BOLD),
            texto("Tabela pequena, com poucas features: usar regressão logística e gradient boosting como modelos principais.",
                  19, COR_TEXTO),
            texto("A transformação em imagem só se justificaria com muitas features correlacionadas, e com os mesmos controles.",
                  19, COR_SUAVE),
        ).arrange(DOWN, buff=0.15).move_to(caixa)
        if rec.width > 12.2:
            rec.scale_to_fit_width(12.2)
        self.play(Create(caixa), FadeIn(rec))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_13_encerramento(self):
        fim = texto("Obrigada!", 84, weight=BOLD)
        sub = texto("Perguntas?", 36, COR_DESTAQUE)
        rodape = texto("Código, resultados e relatórios no repositório do projeto", 20, COR_SUAVE)
        VGroup(fim, sub, rodape).arrange(DOWN, buff=0.45)
        self.play(Write(fim))
        self.play(FadeIn(sub, shift=UP * 0.2))
        self.play(FadeIn(rodape))
        self.next_slide()
