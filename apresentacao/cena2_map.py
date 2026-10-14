"""
CENA 2 — Preparação dos dados e etapa MAP, com as fórmulas de cada critério
==========================================================================

Roteiro (cada método slide_... termina num `self.next_slide()`, onde a apresentação pausa):
   1. Título
   2. A tabela é dividida em treino (80%) e teste (20%), estratificado
   3. Um paciente do treino: 30 quadrados, "feature k" em cima, valor bruto dentro
   4. Padronização z-score: a fórmula e o cálculo com os valores dos quadrados
   5. Todos os valores viram z (animação) e os quadrados viram tons de cinza
   6. O Map usa a COLUNA inteira: gráfico de pontos da feature 28 (455 pacientes)
   7. Alternativa 1: distância euclidiana (fórmula, cálculo, saída)
   8. Alternativa 2: ANOVA F
   9. Alternativa 3: correlação de Pearson
  10. Alternativa 4: Fisher score
  11. Alternativa 5: Gain Ratio (faixas + fórmulas)
  12. Alternativa 6: informação mútua
  13. Alternativa 7: Random Forest (Gini)
  14. Tabela-resumo dos critérios, com destaque para o escolhido
  15. O ranking do critério escolhido em barras horizontais

COMO RODAR (de dentro da pasta apresentacao/):
    .venv/bin/manim-slides render -ql cena2_map.py Cena2Map    # prévia rápida
    .venv/bin/manim-slides render cena2_map.py Cena2Map        # 1080p
    .venv/bin/manim-slides present Cena2Map                    # apresentar

Todos os números são REAIS (dados/mol_dados.json, gerado por gerar_dados.py).
Fórmulas usam LaTeX: formula(r"...") — o r antes das aspas evita problemas com a barra \\.
Cores, fonte e funções comuns ficam em estilo.py.
"""
import numpy as np
from manim import *

from estilo import (COR_BENIGNO, COR_BOM, COR_DESTAQUE, COR_FUNDO, COR_MALIGNO, COR_RUIM,
                    COR_SUAVE, COR_TEXTO, DADOS, GRADIENTE_RANKING, N_FEATURES, CenaBase,
                    br, cor_do_pixel, formula, nome_curto, pixel, tex, texto, valor_bruto)

NOMES = DADOS["feature_names"]
F_TOP, F_LOW = DADOS["f_top"], DADOS["f_low"]        # índices 0..29 (feature 28 e feature 10)
Y_COL = np.array(DADOS["colunas"]["y"])               # rótulos dos 455 pacientes do treino
N_B, N_M = DADOS["contagens"]["treino"]
N = N_B + N_M

# ---- Geometria do gráfico de pontos (slides 6 a 13) ----
EIXO_MIN, EIXO_MAX = -2.5, 3.5    # faixa de z mostrada (valores fora são encostados na borda)
Y_BENIGNO, Y_MALIGNO, Y_EIXO = 2.15, 1.2, 0.5
ESPALHAMENTO = 0.3                # quanto os pontos se espalham na vertical (só visual)

# Nomes dos critérios como aparecem nos slides, na ordem do DADOS["scores"]
CRITERIOS = [
    ("euclidean", "Distância euclidiana"),
    ("anova", "ANOVA F"),
    ("pearson", "Pearson |r|"),
    ("fisher", "Fisher score"),
    ("gain_ratio", "Gain Ratio"),
    ("mutual_info", "Informação mútua"),
    ("random_forest", "Random Forest"),
]


class Cena2Map(CenaBase):
    def construct(self):
        self.slide_1_titulo()
        self.slide_2_treino_teste()
        self.slide_3_um_paciente()
        self.slide_4_formula_zscore()
        self.slide_5_aplica_zscore()
        self.slide_6_coluna_inteira()
        self.slide_7_euclidiana()
        self.slide_8_anova()
        self.slide_9_pearson()
        self.slide_10_fisher()
        self.slide_11_gain_ratio()
        self.slide_12_informacao_mutua()
        self.slide_13_random_forest()
        self.slide_14_tabela_resumo()
        self.slide_15_barras()

    # =================================================================
    # Ajudantes
    # =================================================================
    def trocar_cabecalho(self, conteudo, tamanho=32, cor=COR_TEXTO):
        """Troca o título do topo com uma animação."""
        novo = self.cabecalho(conteudo, tamanho, cor)
        if getattr(self, "titulo", None) is not None and self.titulo in self.mobjects:
            self.play(ReplacementTransform(self.titulo, novo))
        else:
            self.play(FadeIn(novo))
        self.titulo = novo

    def trocar_area(self, *mobs):
        """
        A metade de baixo da tela (fórmulas e contas) é trocada a cada critério.
        Esta função tira o que estava lá e devolve o novo grupo.
        """
        if getattr(self, "area", None) is not None:
            self.play(FadeOut(self.area))
        self.area = VGroup(*mobs)
        return self.area

    # =================================================================
    def slide_1_titulo(self):
        etapa = texto("ETAPA 1", 30, COR_DESTAQUE)
        titulo = texto("MAP", 96, weight=BOLD)
        sub = texto("Ordenar as features pela importância", 32, COR_SUAVE)
        VGroup(etapa, titulo, sub).arrange(DOWN, buff=0.35)
        self.play(FadeIn(etapa, shift=DOWN * 0.3))
        self.play(Write(titulo))
        self.play(FadeIn(sub, shift=UP * 0.3))
        self.next_slide()
        self.limpar()

    # =================================================================
    def slide_2_treino_teste(self):
        self.trocar_cabecalho("Antes de tudo: separar treino e teste")

        # Cada faixa horizontal representa ~10 pacientes, colorida pelo diagnóstico.
        total_b, total_m = DADOS["contagens"]["total"]
        tr_b, tr_m = DADOS["contagens"]["treino"]
        te_b, te_m = DADOS["contagens"]["teste"]
        fb, fm = round(total_b / 10), round(total_m / 10)      # 36 e 21 faixas
        fb_tr, fm_tr = round(tr_b / 10), round(tr_m / 10)      # 28/29 e 17 faixas no treino

        rng = np.random.default_rng(3)
        classes = rng.permutation([0] * fb + [1] * fm)
        altura = 0.075
        faixas = VGroup(*[
            Rectangle(width=2.6, height=altura, stroke_width=0)
            .set_fill(COR_BENIGNO if c == 0 else COR_MALIGNO, opacity=0.9)
            for c in classes
        ]).arrange(DOWN, buff=0.01).move_to(DOWN * 0.3)
        rot_total = texto(f"{total_b + total_m} pacientes", 26).next_to(faixas, UP, buff=0.25)
        legenda = VGroup(
            Square(0.2).set_fill(COR_BENIGNO, 1).set_stroke(width=0), texto(f"benigno ({total_b})", 18, COR_BENIGNO),
            Square(0.2).set_fill(COR_MALIGNO, 1).set_stroke(width=0), texto(f"maligno ({total_m})", 18, COR_MALIGNO),
        ).arrange(RIGHT, buff=0.12).next_to(faixas, DOWN, buff=0.3)
        nota = texto("cada faixa ≈ 10 pacientes", 16, COR_SUAVE).next_to(legenda, DOWN, buff=0.12)

        self.play(LaggedStart(*[FadeIn(f) for f in faixas], lag_ratio=0.02), FadeIn(rot_total))
        self.play(FadeIn(legenda), FadeIn(nota))
        self.next_slide()

        # Separação estratificada: as primeiras faixas de cada classe vão para o treino
        idx_b = [i for i, c in enumerate(classes) if c == 0]
        idx_m = [i for i, c in enumerate(classes) if c == 1]
        treino_idx = sorted(idx_b[:fb_tr] + idx_m[:fm_tr])
        teste_idx = sorted(idx_b[fb_tr:] + idx_m[fm_tr:])

        def empilhar(indices, centro):
            alvo = VGroup(*[faixas[i].copy() for i in indices]).arrange(DOWN, buff=0.01).move_to(centro)
            return [(faixas[i], alvo[k].get_center()) for k, i in enumerate(indices)]

        movimentos = empilhar(treino_idx, LEFT * 3.2 + DOWN * 0.1) + empilhar(teste_idx, RIGHT * 3.4 + DOWN * 0.1)
        self.play(FadeOut(VGroup(rot_total, legenda, nota)))
        self.play(*[f.animate.move_to(p) for f, p in movimentos], run_time=2)

        treino = VGroup(*[faixas[i] for i in treino_idx])
        teste = VGroup(*[faixas[i] for i in teste_idx])
        rot_tr = VGroup(
            texto("TREINO · 80%", 28, COR_TEXTO, weight=BOLD),
            texto(f"{tr_b + tr_m} pacientes", 22),
            texto(f"{tr_b} benignos · {tr_m} malignos", 18, COR_SUAVE),
        ).arrange(DOWN, buff=0.1).next_to(treino, RIGHT, buff=0.5)
        rot_te = VGroup(
            texto("TESTE · 20%", 28, COR_TEXTO, weight=BOLD),
            texto(f"{te_b + te_m} pacientes", 22),
            texto(f"{te_b} benignos · {te_m} malignos", 18, COR_SUAVE),
        ).arrange(DOWN, buff=0.1).next_to(teste, UP, buff=0.4)
        self.play(FadeIn(rot_tr), FadeIn(rot_te))

        rodape = VGroup(
            texto("Estratificado: ~37% de malignos nos dois conjuntos.", 22),
            texto("O teste fica guardado até a avaliação final: o MAP usa só o treino.", 22, COR_DESTAQUE),
        ).arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(rodape), teste.animate.set_opacity(0.3))
        self.next_slide()

        self.treino_faixas = treino
        self.treino_classes = [classes[i] for i in treino_idx]
        self.split_extras = VGroup(teste, rot_tr, rot_te, rodape)

    # =================================================================
    def slide_3_um_paciente(self):
        self.trocar_cabecalho("Um paciente do treino: 30 features")

        x = DADOS["paciente"]["x"]
        y_pac = DADOS["paciente"]["y"]
        lado, buff_h, buff_v = 0.95, 0.1, 0.42

        # 30 quadrados em 3 linhas de 10: média, erro padrão e pior valor de cada medida
        self.quadros, self.valores, self.rotulos_feat = VGroup(), VGroup(), VGroup()
        for j in range(N_FEATURES):
            q = Square(side_length=lado, stroke_color=COR_SUAVE, stroke_width=1.5).set_fill(COR_FUNDO, 1)
            self.quadros.add(q)
        self.quadros.arrange_in_grid(rows=3, cols=10, buff=(buff_h, buff_v)).move_to(RIGHT * 0.65 + DOWN * 0.15)
        for j, q in enumerate(self.quadros):
            v = texto(valor_bruto(x[j]), 17).move_to(q)
            if v.width > lado - 0.1:
                v.scale_to_fit_width(lado - 0.1)
            self.valores.add(v)
            self.rotulos_feat.add(texto(f"feature {j + 1}", 13, COR_SUAVE).next_to(q, UP, buff=0.07))

        grupos = VGroup(*[
            texto(nome, 18, COR_SUAVE).next_to(self.quadros[10 * k], LEFT, buff=0.3)
            for k, nome in enumerate(["média", "erro\npadrão", "pior\nvalor"])
        ])

        # A faixa de um paciente maligno sai do bloco de treino e vira os quadrados
        malignas = [f for f, c in zip(self.treino_faixas, self.treino_classes) if c == 1]
        faixa = malignas[5]
        self.play(Indicate(faixa, color=COR_DESTAQUE, scale_factor=1.3))
        self.play(FadeOut(self.split_extras), FadeOut(VGroup(*[f for f in self.treino_faixas if f is not faixa])))
        self.play(ReplacementTransform(faixa, self.quadros), run_time=1.5)
        self.play(LaggedStart(*[FadeIn(r) for r in self.rotulos_feat], lag_ratio=0.02),
                  LaggedStart(*[FadeIn(v) for v in self.valores], lag_ratio=0.02), FadeIn(grupos))

        diag = texto("diagnóstico: maligno" if y_pac == 1 else "diagnóstico: benigno", 20,
                     COR_MALIGNO if y_pac == 1 else COR_BENIGNO).next_to(self.titulo, DOWN, buff=0.15)
        self.play(FadeIn(diag))
        self.grupos_linhas = grupos
        self.next_slide()

        # As escalas são muito diferentes: destaca o maior e o menor valor
        j_max, j_min = int(np.argmax(x)), int(np.argmin(x))
        marcas = VGroup(*[SurroundingRectangle(self.quadros[j], color=COR_RUIM, buff=0.04) for j in (j_max, j_min)])
        aviso = texto(f"Escalas muito diferentes: feature {j_max + 1} vale {valor_bruto(x[j_max])}, "
                      f"feature {j_min + 1} vale {valor_bruto(x[j_min])}", 22, COR_RUIM).to_edge(DOWN, buff=0.35)
        self.play(Create(marcas), FadeIn(aviso))
        self.next_slide()
        self.play(FadeOut(VGroup(marcas, aviso, diag)))

    # =================================================================
    def slide_4_formula_zscore(self):
        self.trocar_cabecalho("Padronização z-score")

        # Sobe e encolhe a grade de quadrados para abrir espaço para as fórmulas
        paciente = VGroup(self.quadros, self.valores, self.rotulos_feat, self.grupos_linhas)
        self.play(paciente.animate.scale(0.78).next_to(self.titulo, DOWN, buff=0.3))

        f_z = formula(r"z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}", tamanho=56)
        f_mu = formula(r"\mu_j = \frac{1}{N}\sum_{i=1}^{N} x_{ij}", tamanho=34)
        f_sig = formula(r"\sigma_j = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (x_{ij}-\mu_j)^2}", tamanho=34)
        defs = VGroup(f_mu, f_sig).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        bloco = VGroup(f_z, defs).arrange(RIGHT, buff=1.2).move_to(DOWN * 1.9)
        nota = texto(f"μ e σ: média e desvio padrão de cada feature nos {N} pacientes do treino",
                     20, COR_SUAVE).to_edge(DOWN, buff=0.3)

        self.play(Write(f_z))
        self.play(FadeIn(defs), FadeIn(nota))
        self.next_slide()

        # Exemplo com os números de dois quadrados (feature 1 e a de maior valor)
        x, mu, sig = DADOS["paciente"]["x"], DADOS["mu"], DADOS["sigma"]
        self.play(FadeOut(VGroup(defs, nota)), f_z.animate.move_to(LEFT * 4.3 + DOWN * 1.9).scale(0.8))
        anterior = None
        for j in [0, int(np.argmax(x))]:
            z = (x[j] - mu[j]) / sig[j]
            marca = SurroundingRectangle(self.quadros[j], color=COR_DESTAQUE, buff=0.04)
            conta = formula(f"z_{{{j + 1}}}", "=", rf"{{{tex(x[j], 2)} - {tex(mu[j], 2)} \over {tex(sig[j], 3)}}}",
                            "=", tex(z, 3), tamanho=46)
            conta[-1].set_color(COR_DESTAQUE)
            conta.move_to(RIGHT * 1.6 + DOWN * 1.9)
            legenda = texto(f"feature {j + 1} ({nome_curto(NOMES[j])}): valor {valor_bruto(x[j])}, "
                            f"μ = {br(mu[j], 2)}, σ = {br(sig[j], 3)}", 20, COR_SUAVE).to_edge(DOWN, buff=0.35)
            if anterior:
                self.play(FadeOut(anterior))
            self.play(Create(marca), FadeIn(legenda))
            self.play(TransformFromCopy(self.valores[j], conta[2]), FadeIn(conta[:2]), run_time=1.2)
            self.play(FadeIn(conta[3:]))
            anterior = VGroup(marca, conta, legenda)
            self.next_slide()
        self.play(FadeOut(anterior), FadeOut(f_z))

    # =================================================================
    def slide_5_aplica_zscore(self):
        self.trocar_cabecalho("Todos os valores padronizados")
        z = DADOS["paciente"]["z"]

        # Cada número bruto vira o seu z; cada quadrado vira um tom de cinza
        novos = VGroup()
        for j, v in enumerate(self.valores):
            cor_texto = "#111111" if z[j] > 0 else COR_TEXTO   # texto escuro em pixel claro
            novos.add(texto(br(z[j], 2), 17, cor_texto).scale(0.78).move_to(v))
        self.play(LaggedStart(*[
            # a ordem importa: o último objeto animado é desenhado por cima
            AnimationGroup(self.quadros[j].animate.set_fill(cor_do_pixel(z[j]), 1),
                           Transform(self.valores[j], novos[j]))
            for j in range(N_FEATURES)
        ], lag_ratio=0.05), run_time=3)

        explicacao = VGroup(
            texto("Agora todas as features estão na mesma escala:", 24),
            texto("0 = média do treino · +1 = um desvio padrão acima · −1 = um abaixo", 22, COR_DESTAQUE),
            texto("Na imagem, o valor vira tom de cinza: escuro = baixo, claro = alto", 20, COR_SUAVE),
        ).arrange(DOWN, buff=0.2).move_to(DOWN * 2.3)
        self.play(FadeIn(explicacao, shift=UP * 0.2))
        self.next_slide()
        self.play(FadeOut(explicacao))

    # =================================================================
    def slide_6_coluna_inteira(self):
        self.trocar_cabecalho("MAP: uma nota de importância para cada feature")
        pergunta = VGroup(
            texto("Quanto cada feature separa benignos de malignos?", 26),
            texto("A nota usa a COLUNA inteira do treino (455 pacientes), não um paciente só.", 22, COR_DESTAQUE),
        ).arrange(DOWN, buff=0.2).move_to(DOWN * 2.3)
        self.play(FadeIn(pergunta))
        self.play(Indicate(self.quadros[F_TOP], color=COR_DESTAQUE, scale_factor=1.25))
        self.next_slide()

        # ---- Gráfico de pontos da feature 28: um ponto por paciente ----
        self.eixo = NumberLine(x_range=[EIXO_MIN, EIXO_MAX, 1], length=10, color=COR_SUAVE,
                               include_numbers=True, font_size=22).move_to([0.6, Y_EIXO, 0])
        rng = np.random.default_rng(0)
        self.jitter = rng.uniform(-ESPALHAMENTO, ESPALHAMENTO, N)

        self.pontos = VGroup(*[
            Dot(radius=0.035, color=COR_BENIGNO if Y_COL[i] == 0 else COR_MALIGNO, fill_opacity=0.75)
            for i in range(N)
        ])
        self.posicionar_pontos("top", animar=False)
        self.rot_linhas = VGroup(
            texto(f"Benigno\n{N_B}", 20, COR_BENIGNO).move_to([-5.6, Y_BENIGNO, 0]),
            texto(f"Maligno\n{N_M}", 20, COR_MALIGNO).move_to([-5.6, Y_MALIGNO, 0]),
        )
        self.rot_feature = texto(f"feature {F_TOP + 1} · {nome_curto(NOMES[F_TOP])} (valor padronizado z)",
                                 20, COR_SUAVE).move_to([0.6, Y_BENIGNO + 0.78, 0])

        quadro = self.quadros[F_TOP].copy()
        self.play(FadeOut(VGroup(pergunta, self.quadros, self.valores, self.rotulos_feat, self.grupos_linhas)),
                  quadro.animate.move_to([0.6, 1.7, 0]))
        self.play(ReplacementTransform(quadro, self.pontos), FadeIn(self.eixo), FadeIn(self.rot_linhas),
                  FadeIn(self.rot_feature), run_time=2)

        self.marcas_media = self.criar_marcas_media("top")
        self.play(FadeIn(self.marcas_media))
        nota = texto("Cada ponto é um paciente. As linhas brancas são as médias de cada classe.",
                     22, COR_SUAVE).move_to(DOWN * 1.2)
        self.play(FadeIn(nota))
        self.area = nota
        self.next_slide()

    def posicionar_pontos(self, qual, animar=True):
        """Coloca os 455 pontos nos valores da coluna 'top' (feature 28) ou 'low' (feature 10)."""
        valores = np.clip(DADOS["colunas"][qual], EIXO_MIN, EIXO_MAX)
        alvos = [
            [self.eixo.n2p(v)[0], (Y_BENIGNO if Y_COL[i] == 0 else Y_MALIGNO) + self.jitter[i], 0]
            for i, v in enumerate(valores)
        ]
        if animar:
            self.play(*[d.animate.move_to(p) for d, p in zip(self.pontos, alvos)], run_time=2)
        else:
            for d, p in zip(self.pontos, alvos):
                d.move_to(p)

    def criar_marcas_media(self, qual):
        """Traço branco na média de cada classe."""
        est = DADOS["estat_top" if qual == "top" else "estat_low"]
        marcas = VGroup()
        for c, y_linha in [("0", Y_BENIGNO), ("1", Y_MALIGNO)]:
            x = self.eixo.n2p(est[c]["media"])[0]
            marcas.add(Line([x, y_linha - 0.4, 0], [x, y_linha + 0.4, 0], color=WHITE, stroke_width=5))
        return marcas

    def trocar_feature(self, qual):
        """Anima os pontos e as médias para outra feature ('top' ou 'low')."""
        j = F_TOP if qual == "top" else F_LOW
        novo_rot = texto(f"feature {j + 1} · {nome_curto(NOMES[j])} (valor padronizado z)",
                         20, COR_SUAVE).move_to(self.rot_feature)
        novas_marcas = self.criar_marcas_media(qual)
        self.play(ReplacementTransform(self.rot_feature, novo_rot))
        self.rot_feature = novo_rot
        valores = np.clip(DADOS["colunas"][qual], EIXO_MIN, EIXO_MAX)
        alvos = [[self.eixo.n2p(v)[0], (Y_BENIGNO if Y_COL[i] == 0 else Y_MALIGNO) + self.jitter[i], 0]
                 for i, v in enumerate(valores)]
        self.play(*[d.animate.move_to(p) for d, p in zip(self.pontos, alvos)],
                  Transform(self.marcas_media, novas_marcas), run_time=2)

    def rotulo_alternativa(self, numero, nome, extra=""):
        self.trocar_cabecalho(f"Alternativa {numero} · {nome}{extra}", 30)

    # =================================================================
    def slide_7_euclidiana(self):
        self.rotulo_alternativa(1, "Distância euclidiana", "  (a melhor no artigo)")
        est = DADOS["estat_top"]
        mb, mm = est["0"]["media"], est["1"]["media"]

        f = formula(r"s_j", "=", r"\left|\, \bar{z}_j^{\,(M)} - \bar{z}_j^{\,(B)} \,\right|", tamanho=50)
        explic = texto("distância entre a média dos malignos (M) e a dos benignos (B)", 20, COR_SUAVE)
        VGroup(f, explic).arrange(DOWN, buff=0.25).move_to(DOWN * 1.15)
        self.trocar_area(f, explic)
        self.play(Write(f), FadeIn(explic))
        self.next_slide()

        # Seta entre as duas médias, no meio das duas linhas de pontos
        y_seta = (Y_BENIGNO + Y_MALIGNO) / 2
        seta = DoubleArrow([self.eixo.n2p(mb)[0], y_seta, 0], [self.eixo.n2p(mm)[0], y_seta, 0],
                           buff=0, color=COR_DESTAQUE, stroke_width=4, tip_length=0.18)
        conta = formula(rf"s_{{{F_TOP + 1}}}", "=", rf"\left|\, {tex(mm, 3)} - ({tex(mb, 3)}) \,\right|",
                        "=", tex(abs(mm - mb), 3), tamanho=46).move_to(DOWN * 2.7)
        conta[-1].set_color(COR_DESTAQUE)
        self.play(GrowFromCenter(seta))
        self.play(Write(conta))
        self.area.add(conta)
        self.next_slide()

        # A mesma conta para a feature 10: as médias quase coincidem
        self.play(FadeOut(seta))
        self.trocar_feature("low")
        est_l = DADOS["estat_low"]
        conta_l = formula(rf"s_{{{F_LOW + 1}}}", "=",
                          rf"\left|\, {tex(est_l['1']['media'], 3)} - ({tex(est_l['0']['media'], 3)}) \,\right|",
                          "=", tex(abs(est_l['1']['media'] - est_l['0']['media']), 3), tamanho=46).move_to(conta)
        conta_l[-1].set_color(COR_RUIM)
        self.play(ReplacementTransform(conta, conta_l))
        self.area.remove(conta)
        self.area.add(conta_l)
        saida = texto("Saída: uma nota s_j para cada uma das 30 features  →  feature 28 bem acima da 10",
                      20, COR_SUAVE).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(saida))
        self.area.add(saida)
        self.next_slide()
        self.trocar_feature("top")

    # =================================================================
    def faixas_dispersao(self):
        """Faixas sombreadas de ±1 desvio padrão em volta da média de cada classe ('dentro')."""
        est = DADOS["estat_top"]
        faixas = VGroup()
        for c, y_linha, cor in [("0", Y_BENIGNO, COR_BENIGNO), ("1", Y_MALIGNO, COR_MALIGNO)]:
            m, dp = est[c]["media"], np.sqrt(est[c]["var"])
            x0, x1 = self.eixo.n2p(m - dp)[0], self.eixo.n2p(m + dp)[0]
            faixas.add(Rectangle(width=x1 - x0, height=0.78, stroke_color=cor, stroke_width=2)
                       .set_fill(cor, 0.12).move_to([(x0 + x1) / 2, y_linha, 0]))
        return faixas

    def setas_entre(self):
        """Setas da média geral (z = 0) até a média de cada classe ('entre')."""
        est = DADOS["estat_top"]
        x0 = self.eixo.n2p(0)[0]
        linha_geral = DashedLine([x0, Y_MALIGNO - 0.5, 0], [x0, Y_BENIGNO + 0.5, 0], color=WHITE, stroke_width=2)
        setas = VGroup(*[
            Arrow([x0, y_linha + 0.42, 0], [self.eixo.n2p(est[c]["media"])[0], y_linha + 0.42, 0],
                  buff=0, color=COR_DESTAQUE, stroke_width=4, tip_length=0.15)
            for c, y_linha in [("0", Y_BENIGNO), ("1", Y_MALIGNO)]
        ])
        return VGroup(linha_geral, setas)

    def numeros_anova(self):
        est = DADOS["estat_top"]
        entre = sum(est[c]["n"] * est[c]["media"] ** 2 for c in "01")       # média geral = 0
        dentro_soma = sum(est[c]["n"] * est[c]["var"] for c in "01")
        return est, entre, dentro_soma

    # =================================================================
    def slide_8_anova(self):
        self.rotulo_alternativa(2, "ANOVA (teste F)")
        est, entre, dentro_soma = self.numeros_anova()
        dentro = dentro_soma / (N - 2)

        f = formula(r"F_j = \frac{\displaystyle \frac{1}{C-1}\sum_{c} n_c \left(\bar{z}_j^{(c)} - \bar{z}_j\right)^2}"
                    r"{\displaystyle \frac{1}{N-C}\sum_{c}\sum_{i \in c} \left(z_{ij} - \bar{z}_j^{(c)}\right)^2}",
                    tamanho=30)
        legenda_f = VGroup(
            texto("em cima: variação ENTRE as classes (setas amarelas)", 16, COR_DESTAQUE),
            texto("embaixo: variação DENTRO das classes (faixas ±1 desvio)", 16, COR_BENIGNO),
            texto("linha tracejada: média geral (0, pois z já está padronizado)", 16, COR_SUAVE),
        ).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        bloco_f = VGroup(f, legenda_f).arrange(DOWN, buff=0.25).move_to(LEFT * 3.6 + DOWN * 2.1)
        self.trocar_area(bloco_f)
        self.play(Write(f), FadeIn(legenda_f))

        entre_vis, faixas = self.setas_entre(), self.faixas_dispersao()
        self.play(FadeIn(entre_vis))
        self.play(FadeIn(faixas))
        self.next_slide()

        contas = VGroup(
            formula(rf"\text{{entre}} = 285 \cdot ({tex(est['0']['media'], 3)})^2 + 170 \cdot {tex(est['1']['media'], 3)}^2"
                    rf" = {tex(entre, 1)}", tamanho=25),
            formula(rf"\text{{dentro}} = \frac{{285 \cdot {tex(est['0']['var'], 3)} + 170 \cdot {tex(est['1']['var'], 3)}}}{{453}}"
                    rf" = {tex(dentro, 3)}", tamanho=25),
            formula(rf"F_{{{F_TOP + 1}}} = \frac{{{tex(entre, 1)}}}{{{tex(dentro, 3)}}}", r"\approx",
                    tex(entre / dentro, 1), tamanho=30),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        contas.move_to(DOWN * 2.1).to_edge(RIGHT, buff=0.35)
        contas[2][-1].set_color(COR_DESTAQUE)
        for c in contas:
            self.play(Write(c), run_time=1.2)
        self.area.add(contas)
        self.next_slide()
        self.play(FadeOut(VGroup(entre_vis, faixas)))

    # =================================================================
    def slide_9_pearson(self):
        self.rotulo_alternativa(3, "Correlação de Pearson")
        est = DADOS["estat_top"]
        r = DADOS["scores"]["pearson"][F_TOP]

        f = formula(r"r_j = \frac{\sum_i (z_{ij}-\bar{z}_j)(y_i-\bar{y})}"
                    r"{\sqrt{\sum_i (z_{ij}-\bar{z}_j)^2}\,\sqrt{\sum_i (y_i-\bar{y})^2}}",
                    r"\qquad s_j = |r_j|", tamanho=36).move_to(DOWN * 1.2)
        self.trocar_area(f)
        self.play(Write(f))

        # y vale 0 para benigno e 1 para maligno: a reta liga as duas médias
        p_b = [self.eixo.n2p(est["0"]["media"])[0], Y_BENIGNO, 0]
        p_m = [self.eixo.n2p(est["1"]["media"])[0], Y_MALIGNO, 0]
        reta = Line(p_b, p_m, color=COR_DESTAQUE, stroke_width=5)
        rot_y = VGroup(texto("y = 0", 18, COR_BENIGNO).move_to([5.9, Y_BENIGNO, 0]),
                       texto("y = 1", 18, COR_MALIGNO).move_to([5.9, Y_MALIGNO, 0]))
        self.play(FadeIn(rot_y), Create(reta))
        self.next_slide()

        conta = formula(rf"r_{{{F_TOP + 1}}}", "=", tex(r, 3), r"\quad\Rightarrow\quad s_{%d} = " % (F_TOP + 1),
                        tex(r, 3), tamanho=42).move_to(DOWN * 2.6)
        conta[-1].set_color(COR_DESTAQUE)
        anova = texto("Com 2 classes: F = (N − 2)·r² / (1 − r²) = 453 · 0,618 / 0,382 ≈ 733  →  mesma ordem da ANOVA",
                      19, COR_SUAVE).to_edge(DOWN, buff=0.3)
        self.play(Write(conta))
        self.play(FadeIn(anova))
        self.area.add(conta, anova)
        self.next_slide()
        self.play(FadeOut(VGroup(reta, rot_y)))

    # =================================================================
    def slide_10_fisher(self):
        self.rotulo_alternativa(4, "Fisher score")
        est, entre, dentro_soma = self.numeros_anova()
        f = formula(r"F^{\text{Fisher}}_j = \frac{\sum_{c} n_c \left(\bar{z}_j^{(c)} - \bar{z}_j\right)^2}"
                    r"{\sum_{c} n_c \left(\sigma_j^{(c)}\right)^2}", tamanho=40).move_to(DOWN * 1.2)
        self.trocar_area(f)
        entre_vis, faixas = self.setas_entre(), self.faixas_dispersao()
        self.play(Write(f), FadeIn(entre_vis), FadeIn(faixas))
        self.next_slide()

        conta = formula(rf"F^{{\text{{Fisher}}}}_{{{F_TOP + 1}}} = \frac{{{tex(entre, 1)}}}{{285 \cdot {tex(est['0']['var'], 3)}"
                        rf" + 170 \cdot {tex(est['1']['var'], 3)}}} = \frac{{{tex(entre, 1)}}}{{{tex(dentro_soma, 1)}}} =",
                        tex(entre / dentro_soma, 3), tamanho=36).move_to(DOWN * 2.6)
        conta[-1].set_color(COR_DESTAQUE)
        nota = texto("É a ANOVA sem os graus de liberdade  →  mesma ordem das features", 19, COR_SUAVE) \
            .to_edge(DOWN, buff=0.3)
        self.play(Write(conta))
        self.play(FadeIn(nota))
        self.area.add(conta, nota)
        self.next_slide()
        self.play(FadeOut(VGroup(entre_vis, faixas)))

    # =================================================================
    def slide_11_gain_ratio(self):
        self.rotulo_alternativa(5, "Gain Ratio (ganho de informação)")
        gr = DADOS["gain_ratio"]
        faixas_gr = gr["faixas"]

        # 1) Divide a feature em 10 faixas com o mesmo número de pacientes
        col = DADOS["colunas"]["top"]
        limites = [min(col)] + [f["limite_sup"] for f in faixas_gr[:-1]] + [max(col)]
        divisorias = VGroup(*[
            DashedLine([self.eixo.n2p(l)[0], Y_MALIGNO - 0.45, 0], [self.eixo.n2p(l)[0], Y_BENIGNO + 0.45, 0],
                       color=COR_TEXTO, stroke_width=1.5, dash_length=0.06)
            for l in limites[1:-1]
        ])
        contagens = VGroup()
        for k, fx in enumerate(faixas_gr):
            centro = (self.eixo.n2p(limites[k])[0] + self.eixo.n2p(limites[k + 1])[0]) / 2
            contagens.add(VGroup(
                texto(str(fx["benignos"]), 16, COR_BENIGNO).move_to([centro, Y_EIXO + 0.05, 0]),
                texto(str(fx["malignos"]), 16, COR_MALIGNO).move_to([centro, Y_EIXO - 0.3, 0]),
            ))
        explic = texto("10 faixas com o mesmo número de pacientes. Embaixo: quantos benignos e malignos em cada.",
                       20, COR_SUAVE).move_to(DOWN * 1.0)
        self.trocar_area(explic)
        self.play(FadeOut(self.eixo.numbers), Create(divisorias))
        self.play(LaggedStart(*[FadeIn(c) for c in contagens], lag_ratio=0.1), FadeIn(explic))
        pura = texto("faixas das pontas quase PURAS → saber a faixa quase revela o diagnóstico",
                     22, COR_DESTAQUE).move_to(DOWN * 1.6)
        self.play(FadeIn(pura))
        self.area.add(pura)
        self.next_slide()

        # 2) As fórmulas, com os números reais
        p_b, p_m = N_B / N, N_M / N
        linhas = VGroup(
            formula(rf"H(Y) = -\left(\tfrac{{{N_B}}}{{{N}}}\ln\tfrac{{{N_B}}}{{{N}}} + \tfrac{{{N_M}}}{{{N}}}\ln\tfrac{{{N_M}}}{{{N}}}\right)"
                    rf" = {tex(gr['H_Y'], 3)}", tamanho=25),
            formula(rf"H(Y \mid B) = \sum_{{b=1}}^{{10}} \tfrac{{n_b}}{{N}}\, H(Y \mid B=b) = {tex(gr['H_Y_dado_B'], 3)}", tamanho=25),
            formula(rf"IG = H(Y) - H(Y \mid B) = {tex(gr['H_Y'], 3)} - {tex(gr['H_Y_dado_B'], 3)} = {tex(gr['IG'], 3)}", tamanho=25),
            formula(rf"\text{{SplitInfo}} = -\sum_{{b}} \tfrac{{n_b}}{{N}}\ln\tfrac{{n_b}}{{N}} = {tex(gr['split_info'], 3)}"
                    r"\;\;(\approx \ln 10)", tamanho=25),
            formula(rf"GR_{{{F_TOP + 1}}} = \frac{{IG}}{{\text{{SplitInfo}}}} = \frac{{{tex(gr['IG'], 3)}}}{{{tex(gr['split_info'], 3)}}} =",
                    tex(DADOS['scores']['gain_ratio'][F_TOP], 3), tamanho=28),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT).move_to(DOWN * 1.95)
        linhas[-1][-1].set_color(COR_DESTAQUE)
        self.trocar_area(linhas)
        for l in linhas:
            self.play(Write(l), run_time=1.0)
        self.next_slide()
        self.play(FadeOut(VGroup(divisorias, contagens)), FadeIn(self.eixo.numbers))

    # =================================================================
    def curvas_densidade(self, qual):
        """Curva suavizada (densidade) de cada classe, desenhada acima do eixo."""
        valores = np.array(DADOS["colunas"][qual])
        grade = np.linspace(EIXO_MIN, EIXO_MAX, 200)
        curvas = VGroup()
        alturas = []
        for c in [0, 1]:
            v = valores[Y_COL == c]
            banda = 1.06 * v.std() * len(v) ** (-1 / 5)       # regra de Silverman
            dens = np.exp(-0.5 * ((grade[:, None] - v[None, :]) / banda) ** 2).sum(1) / (len(v) * banda * np.sqrt(2 * np.pi))
            alturas.append(dens)
        escala = 1.9 / max(a.max() for a in alturas)
        for dens, cor in zip(alturas, [COR_BENIGNO, COR_MALIGNO]):
            pontos = [[self.eixo.n2p(g)[0], Y_EIXO + 0.08 + d * escala, 0] for g, d in zip(grade, dens)]
            curva = VMobject(stroke_color=cor, stroke_width=4).set_points_smoothly(pontos)
            preench = Polygon(*([[pontos[0][0], Y_EIXO + 0.08, 0]] + pontos + [[pontos[-1][0], Y_EIXO + 0.08, 0]]),
                              stroke_width=0).set_fill(cor, 0.25)
            curvas.add(VGroup(preench, curva))
        return curvas

    def slide_12_informacao_mutua(self):
        self.rotulo_alternativa(6, "Informação mútua")
        f = formula(r"I(Z_j; Y) = \sum_{y \in \{B, M\}} \int p(z, y)\, \ln \frac{p(z, y)}{p(z)\, p(y)}\, dz",
                    tamanho=40).move_to(DOWN * 1.2)
        self.trocar_area(f)

        curvas = self.curvas_densidade("top")
        self.play(self.pontos.animate.set_opacity(0.12), FadeOut(self.marcas_media), Write(f))
        self.play(Create(curvas), run_time=1.5)
        explic = texto("Quanto MENOS as curvas se sobrepõem, MAIS a feature informa sobre o diagnóstico.",
                       21, COR_DESTAQUE).move_to(DOWN * 2.3)
        self.play(FadeIn(explic))
        self.area.add(explic)
        self.next_slide()

        conta = VGroup(
            formula(rf"I_{{{F_TOP + 1}}} \approx", tex(DADOS['scores']['mutual_info'][F_TOP], 3),
                    r"\text{ nats}", tamanho=40),
            texto("estimada pelos 3 vizinhos mais próximos de cada ponto (Kraskov)\n"
                  f"quase igual ao ganho de informação do Gain Ratio ({br(DADOS['gain_ratio']['IG'], 3)})",
                  18, COR_SUAVE, line_spacing=1.1),
        ).arrange(DOWN, buff=0.2).move_to(DOWN * 2.95)
        conta[0][1].set_color(COR_DESTAQUE)
        self.play(FadeOut(explic), FadeIn(conta))
        self.area.remove(explic)
        self.area.add(conta)
        self.next_slide()

        # Feature 10: as curvas quase coincidem
        curvas_low = self.curvas_densidade("low")
        j = F_LOW
        novo_rot = texto(f"feature {j + 1} · {nome_curto(NOMES[j])} (valor padronizado z)",
                         20, COR_SUAVE).move_to(self.rot_feature)
        conta_low = formula(rf"I_{{{F_LOW + 1}}} \approx", tex(DADOS['scores']['mutual_info'][F_LOW], 3),
                            r"\text{ nats}", tamanho=40).move_to(conta[0])
        conta_low[1].set_color(COR_RUIM)
        self.play(Transform(curvas, curvas_low), ReplacementTransform(self.rot_feature, novo_rot),
                  ReplacementTransform(conta[0], conta_low), run_time=2)
        self.rot_feature = novo_rot
        self.area.remove(conta[0])
        self.area.add(conta_low)
        self.next_slide()

        # Volta para a feature 28 e para os pontos
        self.play(FadeOut(curvas))
        self.posicionar_pontos("top", animar=False)
        self.marcas_media = self.criar_marcas_media("top")
        rot = texto(f"feature {F_TOP + 1} · {nome_curto(NOMES[F_TOP])} (valor padronizado z)",
                    20, COR_SUAVE).move_to(self.rot_feature)
        self.play(self.pontos.animate.set_opacity(0.75), FadeIn(self.marcas_media),
                  ReplacementTransform(self.rot_feature, rot))
        self.rot_feature = rot

    # =================================================================
    def slide_13_random_forest(self):
        self.rotulo_alternativa(7, "Random Forest", "  (extra, fora do artigo)")
        g = DADOS["gini_split"]

        f = formula(r"\text{Gini}(t) = 1 - \sum_{c} p_{c,t}^2", r"\qquad",
                    r"\Delta = \text{Gini}(\text{pai}) - \sum_{\text{filhos}} \tfrac{n_{\text{filho}}}{N}\,\text{Gini}(\text{filho})",
                    tamanho=34).move_to(DOWN * 1.15)
        self.trocar_area(f)
        self.play(Write(f))

        # Uma divisão de árvore: o melhor limiar para a feature 28
        x_lim = self.eixo.n2p(g["limiar"])[0]
        corte = Line([x_lim, Y_MALIGNO - 0.55, 0], [x_lim, Y_BENIGNO + 0.55, 0], color=COR_DESTAQUE, stroke_width=5)
        rot_corte = texto(f"z ≤ {br(g['limiar'], 2)} ?", 20, COR_DESTAQUE).next_to(corte, UP, buff=0.05)
        lado_e = texto(f"{g['esq'][0]} B · {g['esq'][1]} M", 20).move_to([x_lim - 1.6, Y_BENIGNO + 0.62, 0])
        lado_d = texto(f"{g['dir'][0]} B · {g['dir'][1]} M", 20).move_to([x_lim + 1.6, Y_BENIGNO + 0.62, 0])
        self.play(FadeOut(self.rot_feature), Create(corte), FadeIn(rot_corte))
        self.play(FadeIn(lado_e), FadeIn(lado_d))
        self.next_slide()

        n_e, n_d = sum(g["esq"]), sum(g["dir"])
        contas = VGroup(
            formula(rf"\text{{Gini(pai)}} = 1 - \left(\tfrac{{{N_B}}}{{{N}}}\right)^2 - \left(\tfrac{{{N_M}}}{{{N}}}\right)^2"
                    rf" = {tex(g['gini_pai'], 3)}", tamanho=28),
            formula(rf"\text{{filhos}} = \tfrac{{{n_e}}}{{{N}}} \cdot {tex(g['gini_esq'], 3)} + \tfrac{{{n_d}}}{{{N}}} \cdot"
                    rf" {tex(g['gini_dir'], 3)} = {tex(g['gini_filhos'], 3)}", tamanho=28),
            formula(rf"\Delta = {tex(g['gini_pai'], 3)} - {tex(g['gini_filhos'], 3)} =", tex(g['delta'], 3), tamanho=32),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to(DOWN * 2.45)
        contas[-1][-1].set_color(COR_DESTAQUE)
        for c in contas:
            self.play(Write(c), run_time=1.0)
        final = texto(f"Soma dos Δ de todas as divisões por esta feature, em 100 árvores, normalizada  →  "
                      f"s = {br(DADOS['scores']['random_forest'][F_TOP], 3)}", 19, COR_SUAVE).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(final))
        self.area.add(contas, final)
        self.next_slide()
        self.extras_rf = VGroup(corte, rot_corte, lado_e, lado_d)

    # =================================================================
    def slide_14_tabela_resumo(self):
        self.limpar()
        self.titulo = None
        self.area = None
        self.trocar_cabecalho("Resultado: cada critério gera um ranking")

        cab = ["Critério", f"feature {F_TOP + 1}", f"feature {F_LOW + 1}", "Top 3 do ranking"]
        larguras = [3.2, 1.9, 1.9, 4.6]
        altura = 0.55

        def linha(valores, cor=COR_TEXTO, tamanho=20, negrito=False):
            celulas = VGroup()
            for v, w in zip(valores, larguras):
                caixa = Rectangle(width=w, height=altura, stroke_color=COR_SUAVE, stroke_width=1)
                t = texto(v, tamanho, cor, weight=BOLD if negrito else NORMAL)
                if t.width > w - 0.15:
                    t.scale_to_fit_width(w - 0.15)
                celulas.add(VGroup(caixa, t.move_to(caixa)))
            return celulas.arrange(RIGHT, buff=0)

        def fmt(c, v):
            return br(v, 1) if v >= 10 else br(v, 3)

        linhas = VGroup(linha(cab, COR_DESTAQUE, 20, True))
        for c, nome in CRITERIOS:
            top3 = ", ".join(f"F{k + 1}" for k in DADOS["top3"][c])
            linhas.add(linha([nome, fmt(c, DADOS["scores"][c][F_TOP]), fmt(c, DADOS["scores"][c][F_LOW]), top3]))
        linhas.arrange(DOWN, buff=0).move_to(UP * 0.2)

        usados = sorted({k for c, _ in CRITERIOS for k in DADOS["top3"][c]})
        legenda = texto("   ".join(f"F{k + 1} = {nome_curto(NOMES[k])}" for k in usados), 17, COR_SUAVE) \
            .next_to(linhas, DOWN, buff=0.35)
        if legenda.width > 13:
            legenda.scale_to_fit_width(13)

        self.play(FadeIn(linhas[0]))
        self.play(LaggedStart(*[FadeIn(l, shift=DOWN * 0.1) for l in linhas[1:]], lag_ratio=0.15))
        self.play(FadeIn(legenda))
        self.next_slide()

        destaque = SurroundingRectangle(linhas[1], color=COR_DESTAQUE, buff=0.05, stroke_width=4)
        escolha = texto("Vamos seguir com a distância euclidiana (melhor resultado no artigo)",
                        24, COR_DESTAQUE).to_edge(DOWN, buff=0.5)
        apagar = [celula[1].animate.set_opacity(0.3) for l in linhas[2:] for celula in l]
        self.play(Create(destaque), FadeIn(escolha), *apagar)
        self.next_slide()
        self.limpar()
        self.titulo = None

    # =================================================================
    def slide_15_barras(self):
        self.trocar_cabecalho("Distância euclidiana: a nota das 30 features")
        notas = np.array(DADOS["scores"]["euclidean"])
        ordem = DADOS["ordem"]
        rank_de = {col: r for r, col in enumerate(ordem)}
        cores = color_gradient([ManimColor(c) for c in GRADIENTE_RANKING], N_FEATURES)

        altura_linha, largura_max, x_barras, topo = 0.205, 4.0, -2.4, 2.95

        def y_pos(p):
            return topo - p * altura_linha

        linhas = VGroup()
        for j in range(N_FEATURES):
            barra = Rectangle(width=max(notas[j] / notas.max() * largura_max, 0.02),
                              height=altura_linha * 0.75, stroke_width=0).set_fill(cores[rank_de[j]], 1)
            barra.move_to([x_barras, y_pos(j), 0], aligned_edge=LEFT)
            rot = texto(f"F{j + 1} · {nome_curto(NOMES[j])}", 13, COR_SUAVE)
            rot.move_to([x_barras - 0.15 - rot.width / 2, y_pos(j), 0])
            linhas.add(VGroup(rot, barra))

        nota_ordem = texto("na ordem original\nda tabela", 22, COR_SUAVE, line_spacing=1.2) \
            .to_edge(RIGHT, buff=0.7).shift(UP * 0.5)
        self.play(LaggedStart(*[FadeIn(l[0]) for l in linhas], lag_ratio=0.02), FadeIn(nota_ordem))
        self.play(LaggedStart(*[GrowFromEdge(l[1], LEFT) for l in linhas], lag_ratio=0.03), run_time=2)
        self.next_slide()

        # Ordena: cada linha desliza para a posição do seu ranking
        self.trocar_cabecalho("Ordenar da maior para a menor nota")
        self.play(FadeOut(nota_ordem))
        self.play(*[linhas[j].animate.shift(UP * (y_pos(rank_de[j]) - y_pos(j))) for j in range(N_FEATURES)],
                  run_time=2.5, rate_func=smooth)
        rankings = VGroup(*[texto(str(r + 1), 13, COR_DESTAQUE).move_to([x_barras + largura_max + 0.35, y_pos(r), 0])
                            for r in range(N_FEATURES)])
        saida = VGroup(
            texto("Saída do MAP:", 24, COR_DESTAQUE),
            texto("a ORDEM das features", 24),
            texto("(posição 1 = pixel 1 da imagem)", 18, COR_SUAVE),
        ).arrange(DOWN, buff=0.15).to_edge(RIGHT, buff=0.4)
        self.play(FadeIn(rankings), FadeIn(saida))
        self.next_slide()

        proxima = texto("Próxima etapa → OPTIMIZE: qual formato e quais features entram na imagem?",
                        22, COR_DESTAQUE).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(proxima, shift=UP * 0.2))
        self.next_slide()
