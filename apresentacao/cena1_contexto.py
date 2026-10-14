"""
CENA 1 — Contextualização: Por que transformar uma tabela em imagens?
=============================================================

Roteiro (cada bloco termina num `self.next_slide()`, onde a apresentação pausa):
  1. Título
  2. A tabela de dados (pacientes reais do dataset)
  3. Por que uma CNN não funciona bem direto na tabela
  4. A ideia do MOL: cada linha vira uma imagem
  5. Separando as imagens em benignas e malignas
  6. A CNN classificando as imagens
  7. As três etapas do MOL (roteiro do resto da apresentação)

COMO RODAR (de dentro da pasta apresentacao/):
    .venv/bin/manim-slides render -ql cena1_contexto.py Cena1Contexto   # prévia rápida
    .venv/bin/manim-slides render cena1_contexto.py Cena1Contexto       # 1080p
    .venv/bin/manim-slides present Cena1Contexto                        # apresentar

Cores, fonte e funções comuns ficam em estilo.py.
Cada slide é um método (slide_1_titulo, slide_2_tabela, ...): para mudar um
slide, procure o método dele. Para tirar um slide, comente a chamada em construct().
"""

import numpy as np
from manim import *

from estilo import (COR_BENIGNO, COR_BOM, COR_DESTAQUE, COR_FUNDO, COR_MALIGNO, COR_RUIM,
                    COR_SUAVE, COR_TEXTO, DADOS, CenaBase, imagem_mol, moldura, pixel,
                    texto, valor_bruto)

# Colunas de feature mostradas na tabela (índices 0..29). As demais viram "…".
COLUNAS_VISIVEIS = [0, 1, 2, 3, 4, 5]
LARGURA_ID, LARGURA_FEAT, LARGURA_RETIC, LARGURA_DIAG = 0.75, 1.15, 0.55, 1.2
ALTURA_LINHA = 0.46


def montar_tabela():
    """
    Monta a tabela com as 8 linhas reais de DADOS['tabela'].
    Retorna um VGroup com atributos auxiliares:
        tab.linhas[i]        -> VGroup das células da linha i (0 = cabeçalho)
        tab.coluna_feat[k]   -> VGroup das células da k-ésima feature visível
    """
    X = DADOS["tabela"]["X"]
    y = DADOS["tabela"]["y"]
    ids = DADOS["tabela"]["linhas"]

    def celula(conteudo, largura, cor=COR_TEXTO, tamanho=17, negrito=False, borda=COR_SUAVE):
        caixa = Rectangle(width=largura, height=ALTURA_LINHA, stroke_color=borda, stroke_width=1)
        t = texto(conteudo, tamanho, cor, weight=BOLD if negrito else NORMAL)
        if t.width > largura - 0.1:
            t.scale_to_fit_width(largura - 0.1)
        return VGroup(caixa, t.move_to(caixa))

    cabecalho = ["ID"] + [f"F{k + 1}" for k in COLUNAS_VISIVEIS] + ["…", "F30", "Diagnóstico"]
    larguras = [LARGURA_ID] + [LARGURA_FEAT] * len(COLUNAS_VISIVEIS) + [LARGURA_RETIC, LARGURA_FEAT, LARGURA_DIAG]

    linhas = VGroup()
    linhas.add(VGroup(*[celula(c, w, COR_DESTAQUE, 17, True) for c, w in zip(cabecalho, larguras)])
               .arrange(RIGHT, buff=0))
    for i in range(len(X)):
        diag = "M" if y[i] == 1 else "B"
        cor_diag = COR_MALIGNO if y[i] == 1 else COR_BENIGNO
        valores = ([f"#{ids[i] + 1}"] + [valor_bruto(X[i][k]) for k in COLUNAS_VISIVEIS]
                   + ["…", valor_bruto(X[i][29]), diag])
        cores = [COR_SUAVE] + [COR_TEXTO] * (len(valores) - 2) + [cor_diag]
        linhas.add(VGroup(*[celula(v, w, c) for v, w, c in zip(valores, larguras, cores)])
                   .arrange(RIGHT, buff=0))
    linhas.arrange(DOWN, buff=0)

    tab = VGroup(linhas)
    tab.linhas = linhas
    tab.coluna_feat = [VGroup(*[linhas[i][1 + k] for i in range(len(linhas))])
                       for k in range(len(COLUNAS_VISIVEIS))]
    return tab


def z_da_linha(i):
    """Valores padronizados (com μ e σ do treino) da linha i da tabela exibida."""
    x = np.array(DADOS["tabela"]["X"][i])
    return (x - np.array(DADOS["mu"])) / np.array(DADOS["sigma"])


class Cena1Contexto(CenaBase):
    def construct(self):
        self.slide_1_titulo()
        self.slide_2_tabela()
        self.slide_3_cnn_na_tabela()
        self.slide_4_linha_vira_imagem()
        self.slide_5_benigno_maligno()
        self.slide_6_cnn_nas_imagens()
        self.slide_7_etapas()

    # -----------------------------------------------------------------
    def slide_1_titulo(self):
        titulo = texto("Map-Optimize-Learn", 72, weight=BOLD)
        sub = texto("Como usar uma CNN em dados tabulares?", 32, COR_DESTAQUE)
        rodape = texto("Classificação de câncer de mama · Breast Cancer Wisconsin", 22, COR_SUAVE)
        VGroup(titulo, sub, rodape).arrange(DOWN, buff=0.4)

        self.play(Write(titulo), run_time=1.5)
        self.play(FadeIn(sub, shift=UP * 0.2))
        self.play(FadeIn(rodape))
        self.next_slide()
        self.limpar()

    # -----------------------------------------------------------------
    def slide_2_tabela(self):
        self.titulo = self.cabecalho("Dado tabular")
        self.tab = montar_tabela().scale(0.95).move_to(DOWN * 0.15)
        legenda = texto("569 pacientes × 30 features (medidas do núcleo celular) + diagnóstico",
                        22, COR_SUAVE).to_edge(DOWN, buff=0.45)
        diag = VGroup(texto("M = maligno", 20, COR_MALIGNO), texto("B = benigno", 20, COR_BENIGNO)) \
            .arrange(RIGHT, buff=0.6).next_to(legenda, UP, buff=0.2)

        self.play(FadeIn(self.titulo))
        self.play(FadeIn(self.tab.linhas[0]))
        self.play(LaggedStart(*[FadeIn(l, shift=DOWN * 0.1) for l in self.tab.linhas[1:]],
                              lag_ratio=0.12))
        self.play(FadeIn(legenda), FadeIn(diag))
        self.legenda_tabela = VGroup(legenda, diag)
        self.next_slide()

    # -----------------------------------------------------------------
    def slide_3_cnn_na_tabela(self):
        novo = self.cabecalho("E se usarmos uma CNN direto na tabela?")
        self.play(ReplacementTransform(self.titulo, novo), FadeOut(self.legenda_tabela))
        self.titulo = novo

        linhas = self.tab.linhas
        explica = texto("A CNN procura padrões LOCAIS: olha um bloco 3×3 de cada vez",
                        24).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(explica))

        def bloco(l0, c0):
            """Retângulo cobrindo 3 linhas × 3 colunas a partir da linha l0 e coluna c0."""
            canto_sup_esq = linhas[l0][c0].get_corner(UL)
            canto_inf_dir = linhas[l0 + 2][c0 + 2].get_corner(DR)
            r = Rectangle(width=canto_inf_dir[0] - canto_sup_esq[0],
                          height=canto_sup_esq[1] - canto_inf_dir[1],
                          stroke_color=COR_DESTAQUE, stroke_width=4)
            r.set_fill(COR_DESTAQUE, opacity=0.15)
            return r.move_to((canto_sup_esq + canto_inf_dir) / 2)

        janela = bloco(1, 1)
        self.play(Create(janela))
        for l0, c0 in [(1, 2), (1, 3), (2, 3), (3, 3)]:
            self.play(Transform(janela, bloco(l0, c0)), run_time=0.6)
        self.next_slide()

        # Os problemas, escritos ABAIXO da tabela, com setas apontando para a janela
        self.play(FadeOut(explica))
        p1 = texto("linhas: mistura 3 pacientes diferentes\n(2 malignos e 1 benigno!)", 21, COR_RUIM)
        p2 = texto("colunas: vizinhas só por acaso", 21, COR_RUIM)
        VGroup(p1, p2).arrange(RIGHT, buff=1.2).to_edge(DOWN, buff=0.4)
        s1 = Arrow(p1.get_top(), janela.get_left() + DOWN * 0.3, buff=0.1, color=COR_RUIM, stroke_width=3)
        s2 = Arrow(p2.get_top(), janela.get_bottom() + RIGHT * 0.6, buff=0.1, color=COR_RUIM, stroke_width=3)
        self.play(FadeIn(p1), GrowArrow(s1))
        self.play(FadeIn(p2), GrowArrow(s2))
        self.next_slide()

        # Problema 3: a ordem das colunas é arbitrária -> embaralhar
        self.play(FadeOut(VGroup(janela, p1, p2, s1, s2)))
        explica = texto("Trocar a ordem das colunas não muda a informação…", 24).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(explica))

        cols = self.tab.coluna_feat
        centros = [c.get_center() for c in cols]
        perm = [3, 0, 5, 1, 4, 2]  # coluna k vai para a posição perm[k]
        self.play(*[cols[k].animate.move_to(centros[perm[k]]) for k in range(len(cols))],
                  run_time=1.5, path_arc=PI / 4)
        conclusao = texto("…mas muda completamente o que a CNN \"vê\".", 24, COR_RUIM)
        conclusao.next_to(explica, UP, buff=0.15)
        self.play(FadeIn(conclusao))
        self.play(*[cols[k].animate.move_to(centros[k]) for k in range(len(cols))],
                  run_time=1.5, path_arc=-PI / 4)
        self.next_slide()

        self.play(FadeOut(VGroup(explica, conclusao)))

    # -----------------------------------------------------------------
    def slide_4_linha_vira_imagem(self):
        novo = self.cabecalho("A ideia do MOL: cada paciente vira uma imagem")
        self.play(ReplacementTransform(self.titulo, novo))
        self.titulo = novo

        # Sobe a tabela para abrir espaço
        self.play(self.tab.animate.scale(0.75).to_edge(UP, buff=1.1))
        linhas = self.tab.linhas
        destaque = SurroundingRectangle(linhas[1], color=COR_DESTAQUE, buff=0.03)
        self.play(Create(destaque))

        # A linha vira uma faixa de 30 pixels (todas as 30 features, padronizadas)
        z = z_da_linha(0)
        faixa = VGroup(*[pixel(z[j], 0.36) for j in range(30)]).arrange(RIGHT, buff=0)
        faixa.move_to(DOWN * 0.6)
        rotulo = texto("as 30 features do paciente #1 como pixels (claro = valor alto)",
                       20, COR_SUAVE).next_to(faixa, DOWN, buff=0.25)
        self.play(TransformFromCopy(linhas[1], faixa), run_time=1.5)
        self.play(FadeIn(rotulo))
        self.next_slide()

        # A faixa "dobra" numa imagem 5 x 6
        img = imagem_mol(z, lado=0.36).move_to(DOWN * 1.6)
        alvo = [img[k] for k in range(30)]
        ordem = DADOS["ordem"]
        self.play(FadeOut(rotulo))
        self.play(LaggedStart(*[faixa[ordem[k]].animate.move_to(alvo[k]) for k in range(30)],
                              lag_ratio=0.04), run_time=2.5)
        legenda = texto("1 linha da tabela  →  1 imagem", 26, COR_DESTAQUE).next_to(faixa, RIGHT, buff=1.0)
        self.play(FadeIn(legenda, shift=LEFT * 0.2))
        self.next_slide()

        self.imagem_exemplo = faixa  # os pixels agora formam a imagem
        self.play(FadeOut(VGroup(destaque, legenda)))

    # -----------------------------------------------------------------
    def slide_5_benigno_maligno(self):
        linhas = self.tab.linhas
        y = DADOS["tabela"]["y"]

        # Gera a imagem de cada uma das 8 linhas, saindo da própria linha
        imagens = []
        for i in range(len(y)):
            img = imagem_mol(z_da_linha(i), lado=0.2)
            imagens.append(img)
        fileira = VGroup(*imagens).arrange(RIGHT, buff=0.35).move_to(DOWN * 1.6)

        self.play(FadeOut(self.imagem_exemplo))
        self.play(LaggedStart(*[TransformFromCopy(linhas[i + 1], imagens[i]) for i in range(len(y))],
                              lag_ratio=0.15), run_time=2.5)
        self.next_slide()

        # Separa: benignos à esquerda, malignos à direita
        novo = self.cabecalho("Imagens de pacientes benignos e malignos")
        self.play(ReplacementTransform(self.titulo, novo), FadeOut(self.tab))
        self.titulo = novo

        benignas = VGroup(*[imagens[i] for i in range(len(y)) if y[i] == 0])
        malignas = VGroup(*[imagens[i] for i in range(len(y)) if y[i] == 1])
        alvo_b = benignas.copy().scale(1.6).arrange_in_grid(rows=2, cols=2, buff=0.4).move_to(LEFT * 3.3 + DOWN * 0.3)
        alvo_m = malignas.copy().scale(1.6).arrange_in_grid(rows=2, cols=2, buff=0.4).move_to(RIGHT * 3.3 + DOWN * 0.3)
        self.play(Transform(benignas, alvo_b), Transform(malignas, alvo_m), run_time=1.8)

        molduras_b = VGroup(*[moldura(m, COR_BENIGNO) for m in benignas])
        molduras_m = VGroup(*[moldura(m, COR_MALIGNO) for m in malignas])
        rot_b = texto("Benigno", 32, COR_BENIGNO).next_to(benignas, UP, buff=0.4)
        rot_m = texto("Maligno", 32, COR_MALIGNO).next_to(malignas, UP, buff=0.4)
        self.play(Create(molduras_b), Create(molduras_m), FadeIn(rot_b), FadeIn(rot_m))

        nota = texto("Dá para ver a diferença a olho nu: as imagens malignas são mais claras no topo.",
                     22, COR_SUAVE).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(nota))
        self.next_slide()

        self.img_benigna = VGroup(benignas[0], molduras_b[0]).copy()
        self.img_maligna = VGroup(malignas[0], molduras_m[0]).copy()
        self.limpar(self.titulo)

    # -----------------------------------------------------------------
    def slide_6_cnn_nas_imagens(self):
        novo = self.cabecalho("Agora sim: a CNN classifica as imagens")
        self.play(ReplacementTransform(self.titulo, novo))
        self.titulo = novo

        # ---- Desenho esquemático da CNN ----
        def pilha(n, lado, cor, deslocamento=0.12):
            """Várias 'folhas' sobrepostas = vários filtros/mapas de ativação."""
            folhas = VGroup(*[
                Square(side_length=lado, stroke_color=cor, stroke_width=2)
                .set_fill(COR_FUNDO, opacity=1).shift((RIGHT + UP) * deslocamento * i)
                for i in range(n)
            ])
            return folhas

        y0 = 0.3
        entrada_pos = LEFT * 5.2 + UP * y0
        conv = pilha(5, 1.5, COR_DESTAQUE).move_to(LEFT * 2.4 + UP * y0)
        pool = pilha(5, 0.9, COR_DESTAQUE).move_to(LEFT * 0.1 + UP * y0)
        densa = VGroup(*[Circle(radius=0.14, stroke_color=COR_TEXTO, stroke_width=2) for _ in range(6)]) \
            .arrange(DOWN, buff=0.18).move_to(RIGHT * 2.0 + UP * y0)
        saida_b = Circle(radius=0.3, stroke_color=COR_BENIGNO, stroke_width=3).move_to(RIGHT * 4.1 + UP * (y0 + 0.6))
        saida_m = Circle(radius=0.3, stroke_color=COR_MALIGNO, stroke_width=3).move_to(RIGHT * 4.1 + UP * (y0 - 0.6))
        rot_sb = texto("Benigno", 24, COR_BENIGNO).next_to(saida_b, RIGHT)
        rot_sm = texto("Maligno", 24, COR_MALIGNO).next_to(saida_m, RIGHT)

        nomes = VGroup(
            texto("imagem", 18, COR_SUAVE).move_to(entrada_pos + DOWN * 1.5),
            texto("convolução\n32 filtros 3×3", 18, COR_SUAVE).next_to(conv, DOWN, buff=0.3),
            texto("max-pooling", 18, COR_SUAVE).next_to(pool, DOWN, buff=0.3),
            texto("camada\ndensa", 18, COR_SUAVE).next_to(densa, DOWN, buff=0.3),
            texto("saída", 18, COR_SUAVE).next_to(VGroup(saida_b, saida_m), DOWN, buff=0.3),
        )
        setas = VGroup(
            Arrow(entrada_pos + RIGHT * 1.0, conv.get_left(), buff=0.15, color=COR_SUAVE, stroke_width=3),
            Arrow(conv.get_right(), pool.get_left(), buff=0.15, color=COR_SUAVE, stroke_width=3),
            Arrow(pool.get_right(), densa.get_left(), buff=0.15, color=COR_SUAVE, stroke_width=3),
        )
        ligacoes = VGroup(*[
            Line(n.get_right(), s.get_left(), stroke_width=1, color=COR_SUAVE)
            for n in densa for s in (saida_b, saida_m)
        ])
        rede = VGroup(conv, pool, densa, saida_b, saida_m, rot_sb, rot_sm, nomes, setas, ligacoes)
        self.play(LaggedStart(*[FadeIn(p) for p in [conv, pool, densa, VGroup(saida_b, saida_m, rot_sb, rot_sm)]],
                              lag_ratio=0.3), FadeIn(setas), FadeIn(ligacoes), FadeIn(nomes))
        self.next_slide()

        def classificar(img_com_moldura, saida, cor):
            """Uma imagem entra, o filtro varre, o sinal atravessa a rede, a saída acende."""
            img = img_com_moldura.copy()
            self.play(img.animate.scale_to_fit_height(1.9).move_to(entrada_pos))
            pixels = img[0]
            # filtro 3x3 deslizando sobre a imagem 5x6 (posições do canto superior esquerdo)
            def janela(l, c):
                cantos = [pixels[l * 6 + c].get_corner(UL), pixels[(l + 2) * 6 + c + 2].get_corner(DR)]
                r = Rectangle(width=cantos[1][0] - cantos[0][0], height=cantos[0][1] - cantos[1][1],
                              stroke_color=COR_DESTAQUE, stroke_width=4)
                return r.move_to((cantos[0] + cantos[1]) / 2)
            jan = janela(0, 0)
            self.play(Create(jan), run_time=0.4)
            for l, c in [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]:
                self.play(Transform(jan, janela(l, c)), run_time=0.3)
            self.play(FadeOut(jan), run_time=0.3)
            # o sinal atravessa a rede
            self.play(ShowPassingFlash(setas.copy().set_color(COR_DESTAQUE).set_stroke(width=6), time_width=0.6),
                      run_time=1.2)
            self.play(Indicate(densa, color=COR_DESTAQUE), run_time=0.6)
            self.play(saida.animate.set_fill(cor, opacity=0.9), run_time=0.6)
            return img

        img1 = classificar(self.img_benigna, saida_b, COR_BENIGNO)
        self.next_slide()
        self.play(FadeOut(img1), saida_b.animate.set_fill(opacity=0))
        img2 = classificar(self.img_maligna, saida_m, COR_MALIGNO)
        nota = texto("Na imagem, os vizinhos de cada pixel são escolhidos de propósito.", 22,
                     COR_BOM).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(nota))
        self.next_slide()
        self.limpar()

    # -----------------------------------------------------------------
    def slide_7_etapas(self):
        pergunta = texto("Mas como transformar uma linha numa BOA imagem?", 34).to_edge(UP, buff=0.8)
        self.play(FadeIn(pergunta))

        def caixa(nome, descricao, cor):
            r = RoundedRectangle(width=3.6, height=2.2, corner_radius=0.2, stroke_color=cor, stroke_width=3)
            n = texto(nome, 36, cor, weight=BOLD)
            d = texto(descricao, 20, COR_TEXTO)
            VGroup(n, d).arrange(DOWN, buff=0.3).move_to(r)
            return VGroup(r, n, d)

        etapas = VGroup(
            caixa("MAP", "em que ORDEM as\nfeatures entram", COR_DESTAQUE),
            caixa("OPTIMIZE", "qual FORMATO e\nquais features", COR_MALIGNO),
            caixa("LEARN", "a CNN aprende\ncom as imagens", COR_BENIGNO),
        ).arrange(RIGHT, buff=0.8).move_to(DOWN * 0.3)
        setas = VGroup(*[Arrow(etapas[i].get_right(), etapas[i + 1].get_left(), buff=0.1, color=COR_SUAVE)
                         for i in range(2)])

        self.play(LaggedStart(*[FadeIn(e, shift=UP * 0.2) for e in etapas], lag_ratio=0.35), FadeIn(setas))
        self.next_slide()

        comeco = texto("Começamos pelo MAP", 28, COR_DESTAQUE).next_to(etapas, DOWN, buff=0.6)
        self.play(etapas[1].animate.set_opacity(0.3), etapas[2].animate.set_opacity(0.3),
                  setas.animate.set_opacity(0.3), Create(SurroundingRectangle(etapas[0], color=COR_DESTAQUE, buff=0.12, corner_radius=0.25)), FadeIn(comeco))
        self.next_slide()
