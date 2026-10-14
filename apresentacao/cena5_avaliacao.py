"""
CENA 5 — Etapa AVALIAÇÃO: o MOL é melhor que os métodos tradicionais?
=====================================================================

Roteiro (cada método slide_... termina num `self.next_slide()`, onde a apresentação pausa):
   1. Título
   2. Por que um único teste não basta
   3. Validação cruzada estratificada de 10 folds (com as notas reais do melhor MOL)
   4. Sem vazamento: tudo é reajustado dentro de cada fold
   5. Grid search e o teste final
   6. As métricas e por que usar a acurácia balanceada
   7. Os baselines e as estratégias S / E / W
   8. Resultado: os 31 modelos ordenados
   9. Wilcoxon passo a passo (melhor MOL contra o melhor baseline)
  10. Comparações do melhor MOL com os demais
  11. O teste final e por que não confiar nele sozinho
  12. Conclusão da avaliação

COMO RODAR (de dentro da pasta apresentacao/):
    .venv/bin/manim-slides render -ql cena5_avaliacao.py Cena5Avaliacao   # prévia rápida
    .venv/bin/manim-slides render cena5_avaliacao.py Cena5Avaliacao       # 1080p
    .venv/bin/manim-slides present Cena5Avaliacao                         # apresentar

Todos os números vêm do experimento principal (results/principal/),
carregados em dados/mol_dados.json (chave "avaliacao") por gerar_dados.py.
"""
import numpy as np
from manim import *

from estilo import (COR_BENIGNO, COR_BOM, COR_DESTAQUE, COR_FUNDO, COR_MALIGNO, COR_RUIM,
                    COR_SUAVE, COR_TEXTO, DADOS, CenaBase, br, formula, tex, texto)

AV = DADOS["avaliacao"]
MODELOS = AV["modelos"]                          # já ordenados pela acurácia na CV
POR_NOME = {m["nome"]: m for m in MODELOS}
MELHOR_MOL, MELHOR_BASE = AV["melhor_mol"], AV["melhor_baseline"]
N_TESTE = DADOS["contagens"]["teste"]            # [benignos, malignos] no teste

COR_ABC, COR_PSO, COR_EPSO, COR_BASE = COR_BOM, COR_MALIGNO, COR_RUIM, COR_BENIGNO
NOME_MAP = {"euclidean": "euclidiana", "anova": "ANOVA", "fisher": "Fisher", "gain_ratio": "gain ratio",
            "mutual_info": "info. mútua"}


def nome_legivel(nome):
    """'MOL-euclidean-abc' -> 'MOL · euclidiana + ABC';  'LR-S' -> 'LR · S'."""
    partes = nome.split("-")
    if partes[0] == "MOL":
        return f"MOL · {NOME_MAP.get(partes[1], partes[1])} + {partes[2].upper()}"
    return f"{partes[0]} · {partes[1]}"


def cor_do_modelo(nome):
    if nome.startswith("MOL"):
        return {"abc": COR_ABC, "pso": COR_PSO, "epso": COR_EPSO}[nome.split("-")[-1]]
    return COR_BASE


def tabela(linhas, larguras, altura=0.5, tamanho=18, cores=None):
    """Tabela simples: lista de linhas (listas de strings). cores[i][j] opcional."""
    grupo = VGroup()
    for i, linha in enumerate(linhas):
        cels = VGroup()
        for j, (v, w) in enumerate(zip(linha, larguras)):
            caixa = Rectangle(width=w, height=altura, stroke_color=COR_SUAVE, stroke_width=1)
            cor = cores[i][j] if cores else (COR_DESTAQUE if i == 0 else COR_TEXTO)
            t = texto(v, tamanho, cor, weight=BOLD if i == 0 else NORMAL)
            if t.width > w - 0.12:
                t.scale_to_fit_width(w - 0.12)
            cels.add(VGroup(caixa, t.move_to(caixa)))
        grupo.add(cels.arrange(RIGHT, buff=0))
    return grupo.arrange(DOWN, buff=0)


class Cena5Avaliacao(CenaBase):
    def construct(self):
        self.slide_1_titulo()
        self.slide_2_um_teste_nao_basta()
        self.slide_3_validacao_cruzada()
        self.slide_4_sem_vazamento()
        self.slide_5_grid_search()
        self.slide_6_metricas()
        self.slide_7_baselines()
        self.slide_8_ranking()
        self.slide_9_wilcoxon()
        self.slide_10_comparacoes()
        self.slide_11_teste_final()
        self.slide_12_conclusao()

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
        etapa = texto("ETAPA 4", 30, COR_DESTAQUE)
        titulo = texto("AVALIAÇÃO", 96, weight=BOLD)
        sub = texto("O MOL é melhor que os métodos tradicionais?", 32, COR_SUAVE)
        VGroup(etapa, titulo, sub).arrange(DOWN, buff=0.35)
        self.play(FadeIn(etapa, shift=DOWN * 0.3))
        self.play(Write(titulo))
        self.play(FadeIn(sub, shift=UP * 0.3))
        self.next_slide()
        self.limpar()

    # =================================================================
    def slide_2_um_teste_nao_basta(self):
        self.trocar_cabecalho("Por que um único teste não basta")
        t = DADOS["learn"]["teste"]
        n_mal = t["tp"] + t["fn"]
        intro = texto(f"No teste da etapa anterior, a rede acertou {t['tp']} dos {n_mal} pacientes malignos.",
                      24).next_to(self.titulo, DOWN, buff=0.5)
        pontos = VGroup(*[Dot(radius=0.13, color=COR_BOM if i < t["tp"] else COR_RUIM) for i in range(n_mal)]) \
            .arrange_in_grid(rows=3, cols=14, buff=0.16).move_to(UP * 0.5)
        sens = formula(rf"\text{{Sensibilidade}} = \frac{{{t['tp']}}}{{{n_mal}}} =", tex(t["tp"] / n_mal, 3), tamanho=40) \
            .next_to(pontos, DOWN, buff=0.5)
        sens[1].set_color(COR_DESTAQUE)
        self.play(FadeIn(intro), LaggedStart(*[FadeIn(p, scale=0.5) for p in pontos], lag_ratio=0.02))
        self.play(Write(sens))
        self.next_slide()

        # Um paciente a mais errado muda bastante a métrica
        novo = formula(rf"\text{{Sensibilidade}} = \frac{{{t['tp'] - 1}}}{{{n_mal}}} =", tex((t["tp"] - 1) / n_mal, 3),
                       tamanho=40).move_to(sens)
        novo[1].set_color(COR_RUIM)
        self.play(pontos[t["tp"] - 1].animate.set_color(COR_RUIM), ReplacementTransform(sens, novo))
        nota = VGroup(
            texto(f"Um único paciente muda a sensibilidade em {br(100 / n_mal, 1)} pontos percentuais.", 24, COR_DESTAQUE),
            texto("Com um teste só, não dá para separar um modelo melhor de um modelo com sorte.", 22, COR_SUAVE),
            texto("Solução: repetir a avaliação várias vezes, em partes diferentes dos dados.", 22, COR_TEXTO),
        ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(nota))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_3_validacao_cruzada(self):
        self.trocar_cabecalho("Validação cruzada estratificada: 10 folds")
        n_tr = sum(DADOS["contagens"]["treino"])
        folds_reais = AV["folds_melhor_mol"]
        larg = 1.12
        blocos = VGroup(*[Rectangle(width=larg, height=0.9, stroke_color=COR_FUNDO, stroke_width=3)
                          .set_fill(COR_BENIGNO, 0.8) for _ in range(10)]).arrange(RIGHT, buff=0).move_to(UP * 1.3)
        numeros = VGroup(*[texto(f"fold {k + 1}", 15, COR_TEXTO).move_to(blocos[k]) for k in range(10)])
        rot = texto(f"os {n_tr} pacientes de treino, divididos em 10 partes com a mesma proporção de malignos",
                    19, COR_SUAVE).next_to(blocos, UP, buff=0.25)
        self.play(LaggedStart(*[FadeIn(b) for b in blocos], lag_ratio=0.05), FadeIn(numeros), FadeIn(rot))

        legenda = VGroup(Square(0.25).set_fill(COR_BENIGNO, 0.8).set_stroke(width=0), texto("treina o modelo", 18, COR_BENIGNO),
                         Square(0.25).set_fill(COR_DESTAQUE, 1).set_stroke(width=0), texto("avalia (nunca visto no treino)", 18, COR_DESTAQUE)) \
            .arrange(RIGHT, buff=0.15).next_to(blocos, DOWN, buff=0.3)
        self.play(FadeIn(legenda))

        rot_notas = texto(f"acurácia balanceada do {nome_legivel(MELHOR_MOL)} em cada fold:", 19, COR_SUAVE) \
            .move_to(DOWN * 0.75)
        notas = VGroup(*[texto(br(v, 3), 20, COR_TEXTO).move_to([blocos[k].get_x(), -1.4, 0])
                         for k, v in enumerate(folds_reais)])
        self.play(FadeIn(rot_notas))
        for k in range(10):
            anims = [blocos[k].animate.set_fill(COR_DESTAQUE, 1), FadeIn(notas[k], shift=DOWN * 0.2)]
            if k > 0:
                anims.append(blocos[k - 1].animate.set_fill(COR_BENIGNO, 0.8))
            self.play(*anims, run_time=0.9 if k < 2 else 0.4)
            if k == 0:
                self.next_slide()
        self.play(blocos[9].animate.set_fill(COR_BENIGNO, 0.8), run_time=0.4)

        media = formula(r"\text{média} \pm \text{desvio} =", rf"{tex(np.mean(folds_reais), 3)} \pm {tex(np.std(folds_reais), 3)}",
                        tamanho=42).move_to(DOWN * 2.5)
        media[1].set_color(COR_DESTAQUE)
        chave = Brace(notas, DOWN, color=COR_SUAVE)
        self.play(GrowFromCenter(chave), Write(media))
        nota = texto("10 notas em vez de 1: dá para ver a média E o quanto ela varia.", 20, COR_SUAVE).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(nota))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_4_sem_vazamento(self):
        self.trocar_cabecalho("Dentro de cada fold, tudo é refeito do zero")

        def caixa(nome, aprende, cor):
            r = RoundedRectangle(width=2.5, height=1.5, corner_radius=0.12, stroke_color=cor, stroke_width=3)
            c = VGroup(texto(nome, 21, cor, weight=BOLD), texto(aprende, 15, COR_TEXTO, line_spacing=1.0)) \
                .arrange(DOWN, buff=0.15).move_to(r)
            return VGroup(r, c)

        etapas = VGroup(
            caixa("Padronização", "aprende μ e σ", COR_SUAVE),
            caixa("Map", "aprende o ranking\ndas features", COR_DESTAQUE),
            caixa("Optimize", "aprende formato\ne máscara", COR_MALIGNO),
            caixa("Learn", "aprende os pesos\nda CNN", COR_BENIGNO),
        ).arrange(RIGHT, buff=0.5).move_to(UP * 0.9)
        setas = VGroup(*[Arrow(etapas[i].get_right(), etapas[i + 1].get_left(), buff=0.05, color=COR_SUAVE, stroke_width=3)
                         for i in range(3)])
        moldura = SurroundingRectangle(etapas, color=COR_BENIGNO, buff=0.3, corner_radius=0.15)
        rot_m = texto("ajustado SÓ com os 9 folds de treino", 20, COR_BENIGNO).next_to(moldura, UP, buff=0.1)
        self.play(LaggedStart(*[FadeIn(e, shift=UP * 0.15) for e in etapas], lag_ratio=0.2), FadeIn(setas))
        self.play(Create(moldura), FadeIn(rot_m))
        self.next_slide()

        val = RoundedRectangle(width=2.6, height=0.8, corner_radius=0.1, stroke_color=COR_DESTAQUE, stroke_width=3) \
            .set_fill(COR_DESTAQUE, 0.2).move_to(LEFT * 5.0 + DOWN * 1.6)
        rot_v = texto("fold de avaliação", 18, COR_DESTAQUE).move_to(val)
        caminho = Arrow(val.get_right(), RIGHT * 3.4 + DOWN * 1.6, buff=0.1, color=COR_DESTAQUE, stroke_width=4)
        rot_c = texto("só PASSA pelas transformações já aprendidas", 18, COR_DESTAQUE).next_to(caminho, UP, buff=0.08)
        nota_f = texto("nota do fold", 20, COR_DESTAQUE).next_to(caminho, RIGHT, buff=0.15)
        self.play(FadeIn(val), FadeIn(rot_v))
        self.play(GrowArrow(caminho), FadeIn(rot_c), FadeIn(nota_f))
        aviso = VGroup(
            texto("Se μ, σ ou o ranking fossem calculados com TODOS os dados, o fold de avaliação já teria", 19, COR_RUIM),
            texto("influenciado o modelo, e a nota sairia otimista. Isso se chama vazamento de dados.", 19, COR_RUIM),
        ).arrange(DOWN, buff=0.1).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(aviso))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_5_grid_search(self):
        self.trocar_cabecalho("Grid search: escolher os hiperparâmetros sem olhar o teste")
        lr_escolhida = POR_NOME[MELHOR_MOL]["melhor_lr"]
        opcoes = [0.1, 0.01, 0.001, 0.0001]
        linhas = VGroup()
        for lr in opcoes:
            escolhida = abs(lr - lr_escolhida) < 1e-12
            cor = COR_DESTAQUE if escolhida else COR_SUAVE
            rot = formula(rf"\eta = {tex(lr, len(str(lr).split('.')[1]))}", tamanho=32, cor=cor)
            mini = VGroup(*[Square(0.22).set_fill(COR_BENIGNO, 0.7).set_stroke(COR_FUNDO, 1) for _ in range(10)]) \
                .arrange(RIGHT, buff=0.03)
            res = texto(f"média dos 10 folds = {br(POR_NOME[MELHOR_MOL]['cv_media'], 3)}  ← a melhor" if escolhida
                        else "média dos 10 folds (menor)", 19, cor)
            linhas.add(VGroup(rot, mini, res))
        for l in linhas:
            l[1].next_to(l[0], RIGHT, buff=0.5)
        linhas.arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        for l in linhas:
            l[0].align_to(linhas, LEFT)
            l[1].set_x(linhas.get_left()[0] + 3.2)
            l[2].next_to(l[1], RIGHT, buff=0.5)
        linhas.move_to(UP * 0.8)
        expl = texto(f"Exemplo: {nome_legivel(MELHOR_MOL)}. Cada taxa de aprendizado é avaliada nos mesmos 10 folds.",
                     19, COR_SUAVE).next_to(self.titulo, DOWN, buff=0.3)
        self.play(FadeIn(expl))
        self.play(LaggedStart(*[FadeIn(l, shift=RIGHT * 0.2) for l in linhas], lag_ratio=0.3))
        self.next_slide()

        # Depois: retreina no treino inteiro e avalia UMA vez no teste
        n_tr, n_te = sum(DADOS["contagens"]["treino"]), sum(N_TESTE)
        b_tr = Rectangle(width=6.4, height=0.7, stroke_width=0).set_fill(COR_BENIGNO, 0.8)
        b_te = Rectangle(width=1.6, height=0.7, stroke_width=0).set_fill(COR_DESTAQUE, 1)
        barra = VGroup(b_tr, b_te).arrange(RIGHT, buff=0.1).move_to(DOWN * 2.1)
        r1 = texto(f"retreina com a melhor configuração no treino inteiro ({n_tr})", 18, COR_TEXTO).move_to(b_tr)
        r2 = texto(f"teste ({n_te})", 18, "#111111").move_to(b_te)
        fim = texto("O teste é usado uma única vez, no final. Ele nunca participa de nenhuma escolha.", 20, COR_DESTAQUE) \
            .to_edge(DOWN, buff=0.35)
        self.play(FadeIn(barra), FadeIn(r1), FadeIn(r2))
        self.play(FadeIn(fim))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_6_metricas(self):
        self.trocar_cabecalho("As métricas (classe positiva = maligno)")

        def cel(sigla, nome, cor):
            r = Rectangle(width=2.5, height=0.95, stroke_color=COR_SUAVE, stroke_width=1.5).set_fill(cor, 0.25)
            return VGroup(r, VGroup(texto(sigla, 26, COR_TEXTO, weight=BOLD), texto(nome, 13, COR_SUAVE))
                          .arrange(DOWN, buff=0.05).move_to(r))

        matriz = VGroup(cel("VP", "maligno previsto maligno", COR_BOM), cel("FN", "maligno previsto benigno", COR_RUIM),
                        cel("FP", "benigno previsto maligno", COR_RUIM), cel("VN", "benigno previsto benigno", COR_BOM)) \
            .arrange_in_grid(rows=2, cols=2, buff=0).move_to(LEFT * 3.8 + UP * 1.0)
        formulas = VGroup(
            formula(r"\text{Sensibilidade} = \frac{VP}{VP + FN}", tamanho=32),
            formula(r"\text{Especificidade} = \frac{VN}{VN + FP}", tamanho=32),
            formula(r"\text{Acurácia balanceada} = \frac{\text{Sens.} + \text{Espec.}}{2}", tamanho=32),
        ).arrange(DOWN, buff=0.55, aligned_edge=LEFT).move_to(RIGHT * 3.0 + UP * 1.0)
        formulas[2].set_color(COR_DESTAQUE)
        notas = VGroup(
            texto("dos malignos, quantos o modelo detecta", 15, COR_SUAVE).next_to(formulas[0], DOWN, buff=0.08).align_to(formulas, LEFT),
            texto("dos benignos, quantos ele libera", 15, COR_SUAVE).next_to(formulas[1], DOWN, buff=0.08).align_to(formulas, LEFT))
        self.play(FadeIn(matriz))
        self.play(LaggedStart(*[Write(f_) for f_ in formulas], lag_ratio=0.4), FadeIn(notas))
        self.next_slide()

        # Por que balanceada: o modelo "todo mundo é benigno"
        nb, nm = N_TESTE
        caso = texto('Um modelo inútil que responde "benigno" para todos os 114 pacientes do teste:', 21, COR_TEXTO) \
            .move_to(DOWN * 1.45)
        contas = VGroup(
            formula(rf"\text{{Acurácia comum}} = \frac{{{nb}}}{{{nb + nm}}} =", tex(nb / (nb + nm), 3), tamanho=30),
            formula(rf"\text{{Acurácia balanceada}} = \frac{{\frac{{0}}{{{nm}}} + \frac{{{nb}}}{{{nb}}}}}{{2}} =", tex(0.5, 3),
                    tamanho=30),
        ).arrange(RIGHT, buff=1.0).move_to(DOWN * 2.35)
        contas[0][1].set_color(COR_RUIM)
        contas[1][1].set_color(COR_DESTAQUE)
        fim = texto("A acurácia comum parece boa (63%). A balanceada mostra a verdade (50% = chute).",
                    20, COR_DESTAQUE).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(caso))
        self.play(Write(contas[0]))
        self.play(Write(contas[1]))
        self.play(FadeIn(fim))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_7_baselines(self):
        self.trocar_cabecalho("Os concorrentes: 6 modelos tradicionais, em até 3 versões")
        defs = VGroup(
            texto("S = sem seleção: usa as 30 features", 19, COR_TEXTO),
            texto("E = embedded: mantém as features que o próprio modelo considera importantes", 19, COR_TEXTO),
            texto("W = wrapper (RFE): treina, remove as menos importantes e repete", 19, COR_TEXTO),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT).next_to(self.titulo, DOWN, buff=0.3)
        self.play(FadeIn(defs))

        familias = [("ANN", "rede neural"), ("LR", "regressão logística"), ("KNN", "vizinhos mais próximos"),
                    ("RF", "Random Forest"), ("XGB", "XGBoost"), ("LGBM", "LightGBM")]
        linhas = [["Modelo", "S", "E", "W"]]
        cores = [[COR_DESTAQUE] * 4]
        for sigla, nome in familias:
            linha, cor_l = [f"{sigla} · {nome}"], [COR_TEXTO]
            vals = {fs: POR_NOME.get(f"{sigla}-{fs}") for fs in "SEW"}
            melhor = max(v["cv_media"] for v in vals.values() if v)
            for fs in "SEW":
                v = vals[fs]
                if v is None:
                    linha.append("não se aplica")
                    cor_l.append(COR_SUAVE)
                else:
                    linha.append(f"{br(v['cv_media'], 3)}  ({v['n_features']} feat.)")
                    cor_l.append(COR_DESTAQUE if v["cv_media"] == melhor else COR_TEXTO)
            linhas.append(linha)
            cores.append(cor_l)
        tab = tabela(linhas, [4.2, 2.7, 2.7, 2.7], altura=0.5, tamanho=18, cores=cores).move_to(DOWN * 1.15)
        self.play(FadeIn(tab[0]))
        self.play(LaggedStart(*[FadeIn(l, shift=DOWN * 0.1) for l in tab[1:]], lag_ratio=0.15))
        nota = texto("Acurácia balanceada média nos 10 folds (número de features usadas). Em amarelo, a melhor versão de cada modelo.\n"
                     "O KNN não tem pesos nem importâncias, por isso só roda em S.", 16, COR_SUAVE, line_spacing=1.1) \
            .to_edge(DOWN, buff=0.2)
        self.play(FadeIn(nota))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_8_ranking(self):
        self.trocar_cabecalho("Resultado: os 31 modelos, do melhor para o pior")
        n = len(MODELOS)
        alt, topo, x0, larg_max = 0.195, 2.95, -3.6, 5.2
        v_min, v_max = 0.88, 1.0

        def larg(v):
            return max((v - v_min) / (v_max - v_min) * larg_max, 0.02)

        linhas = VGroup()
        for i, m in enumerate(MODELOS):
            y = topo - i * alt
            cor = cor_do_modelo(m["nome"])
            barra = Rectangle(width=larg(m["cv_media"]), height=alt * 0.75, stroke_width=0).set_fill(cor, 0.95)
            barra.move_to([x0, y, 0], aligned_edge=LEFT)
            rot = texto(nome_legivel(m["nome"]), 12, cor if m["familia"] == "MOL" else COR_SUAVE)
            rot.move_to([x0 - 0.15 - rot.width / 2, y, 0])
            val = texto(br(m["cv_media"], 3), 12, COR_TEXTO).next_to(barra, RIGHT, buff=0.08)
            linhas.add(VGroup(rot, barra, val))
        eixo_nota = texto("acurácia balanceada média nos 10 folds (o eixo começa em 0,88)", 15, COR_SUAVE) \
            .move_to([x0 + larg_max / 2, topo - n * alt - 0.1, 0])
        self.play(LaggedStart(*[FadeIn(l[0]) for l in linhas], lag_ratio=0.02))
        self.play(LaggedStart(*[GrowFromEdge(l[1], LEFT) for l in linhas], lag_ratio=0.03), run_time=2.5)
        self.play(FadeIn(VGroup(*[l[2] for l in linhas])), FadeIn(eixo_nota))

        legenda = VGroup(*[VGroup(Square(0.2).set_fill(c, 1).set_stroke(width=0), texto(t_, 17, c)).arrange(RIGHT, buff=0.12)
                           for c, t_ in [(COR_BASE, "métodos tradicionais"), (COR_ABC, "MOL com ABC"),
                                         (COR_PSO, "MOL com PSO"), (COR_EPSO, "MOL com EPSO")]]) \
            .arrange(DOWN, buff=0.15, aligned_edge=LEFT).move_to(RIGHT * 4.7 + UP * 2.2)
        self.play(FadeIn(legenda))
        self.next_slide()

        pos_mol = [i for i, m in enumerate(MODELOS) if m["nome"] == MELHOR_MOL][0] + 1
        n_abaixo_base = sum(1 for m in MODELOS[pos_mol:] if m["familia"] != "MOL")
        resumo = VGroup(
            texto(f"Melhor tradicional: {nome_legivel(MELHOR_BASE)}", 19, COR_BASE),
            texto(br(POR_NOME[MELHOR_BASE]["cv_media"], 3), 30, COR_BASE, weight=BOLD),
            texto("Melhor MOL: euclidiana + ABC", 19, COR_ABC),
            texto(br(POR_NOME[MELHOR_MOL]["cv_media"], 3), 30, COR_ABC, weight=BOLD),
            texto(f"({pos_mol}º lugar de {n}; só {n_abaixo_base} tradicionais\nficam abaixo dele)", 16, COR_SUAVE, line_spacing=1.0),
        ).arrange(DOWN, buff=0.15).move_to(RIGHT * 4.7 + DOWN * 0.9)
        destaque = VGroup(SurroundingRectangle(linhas[0], color=COR_BASE, buff=0.03, stroke_width=2),
                          SurroundingRectangle(linhas[pos_mol - 1], color=COR_ABC, buff=0.03, stroke_width=2))
        self.play(Create(destaque), FadeIn(resumo))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_9_wilcoxon(self):
        self.trocar_cabecalho("A diferença é real ou é acaso? Teste de Wilcoxon")
        w = AV["wilcoxon_exemplo"]
        intro = texto(f"Mesmos 10 folds para os dois modelos: A = {nome_legivel(w['A'])}  ·  B = {nome_legivel(w['B'])}",
                      19, COR_SUAVE).next_to(self.titulo, DOWN, buff=0.25)
        self.play(FadeIn(intro))

        d = np.array(w["d"])
        cab = ["fold"] + [str(k + 1) for k in range(10)]
        l_a = ["A (MOL)"] + [br(v, 3) for v in w["a"]]
        l_b = ["B (ANN)"] + [br(v, 3) for v in w["b"]]
        l_d = ["d = A − B"] + [("+" if v > 0 else "") + br(v, 3) for v in d]
        l_p = ["posto de |d|"] + [br(v, 1) for v in w["postos"]]
        cor_d = [COR_DESTAQUE] + [COR_BOM if v > 0 else COR_RUIM if v < 0 else COR_SUAVE for v in d]
        cores = [[COR_DESTAQUE] * 11, [COR_ABC] + [COR_TEXTO] * 10, [COR_BASE] + [COR_TEXTO] * 10, cor_d, cor_d]
        tab = tabela([cab, l_a, l_b, l_d, l_p], [1.9] + [1.02] * 10, altura=0.5, tamanho=17, cores=cores).move_to(UP * 0.85)
        self.play(FadeIn(tab[0]), FadeIn(tab[1]), FadeIn(tab[2]))
        self.next_slide()

        passo1 = texto("1. diferença em cada fold (verde: MOL melhor · vermelho: ANN melhor · cinza: empate)", 18, COR_TEXTO) \
            .next_to(tab, DOWN, buff=0.25)
        self.play(FadeIn(tab[3]), FadeIn(passo1))
        self.next_slide()
        passo2 = texto("2. ordena os |d| do menor (posto 1) ao maior (posto 10); empates dividem o posto", 18, COR_TEXTO) \
            .next_to(passo1, DOWN, buff=0.12)
        self.play(FadeIn(tab[4]), FadeIn(passo2))
        self.next_slide()

        contas = VGroup(
            formula(r"W^{+} = \sum_{d_k > 0} \text{posto}", "=", tex(w["w_mais"], 1), tamanho=32),
            formula(r"W^{-} = \sum_{d_k < 0} \text{posto}", "=", tex(w["w_menos"], 1), tamanho=32),
            formula(r"T = \min(W^{+}, W^{-}) =", tex(min(w["w_mais"], w["w_menos"]), 1), r"\;\Rightarrow\; p =",
                    tex(w["p"], 3), tamanho=32),
        ).arrange(RIGHT, buff=0.7).move_to(DOWN * 2.2)
        contas[0][2].set_color(COR_BOM)
        contas[1][2].set_color(COR_RUIM)
        contas[2][3].set_color(COR_DESTAQUE)
        if contas.width > 12.8:
            contas.scale_to_fit_width(12.8)
        nota_z = texto("(o posto de cada empate é dividido meio a meio entre W⁺ e W⁻)", 15, COR_SUAVE) \
            .next_to(contas, UP, buff=0.12)
        self.play(Write(contas[0]), Write(contas[1]), FadeIn(nota_z))
        self.play(Write(contas[2]))
        veredito = texto(f"p = {br(w['p'], 3)} > 0,05: a vantagem da ANN sobre o MOL NÃO é estatisticamente significativa.",
                         21, COR_DESTAQUE).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(veredito))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_10_comparacoes(self):
        self.trocar_cabecalho(f"O melhor MOL ({br(POR_NOME[MELHOR_MOL]['cv_media'], 3)}) contra os demais")
        linhas = [["Comparado com", "Acurácia (CV)", "MOL − outro", "p-valor", "Significativo?"]]
        cores = [[COR_DESTAQUE] * 5]
        for c in AV["comparacoes"]:
            sig = c["p"] < 0.05
            cor = cor_do_modelo(c["nome"])
            linhas.append([nome_legivel(c["nome"]), br(c["cv_media"], 3), ("+" if c["diferenca"] > 0 else "") + br(c["diferenca"], 3),
                           br(c["p"], 3), "sim" if sig else "não"])
            cores.append([cor, COR_TEXTO, COR_BOM if c["diferenca"] > 0 else COR_RUIM, COR_TEXTO,
                          COR_DESTAQUE if sig else COR_SUAVE])
        tab = tabela(linhas, [4.0, 2.2, 2.2, 1.8, 2.3], altura=0.52, tamanho=19, cores=cores).move_to(UP * 0.35)
        self.play(FadeIn(tab[0]))
        self.play(LaggedStart(*[FadeIn(l, shift=DOWN * 0.1) for l in tab[1:]], lag_ratio=0.12))
        self.next_slide()
        conclusoes = VGroup(
            texto("O MOL fica ABAIXO de todos os tradicionais, mas nenhuma dessas diferenças é significativa.", 20, COR_TEXTO),
            texto("As únicas diferenças significativas são dentro do próprio MOL: ABC melhor que PSO e EPSO.", 20, COR_DESTAQUE),
        ).arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(conclusoes))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_11_teste_final(self):
        self.trocar_cabecalho("E no teste final (114 pacientes)?")
        escolhidos = [MELHOR_BASE, "LR-S", MELHOR_MOL, "MOL-euclidean-pso"]
        linhas = [["Modelo", "CV (10 folds)", "Teste", "Erros no teste", "Malignos perdidos"]]
        cores = [[COR_DESTAQUE] * 5]
        for nome in escolhidos:
            m = POR_NOME[nome]
            linhas.append([nome_legivel(nome), f"{br(m['cv_media'], 3)} ± {br(m['cv_desvio'], 3)}", br(m["teste"], 3),
                           f"{m['fp'] + m['fn']} de 114", f"{m['fn']} de {m['tp'] + m['fn']}"])
            cores.append([cor_do_modelo(nome)] + [COR_TEXTO] * 4)
        tab = tabela(linhas, [4.0, 2.8, 1.6, 2.4, 2.6], altura=0.6, tamanho=20, cores=cores).move_to(UP * 0.9)
        self.play(FadeIn(tab[0]))
        self.play(LaggedStart(*[FadeIn(l, shift=DOWN * 0.1) for l in tab[1:]], lag_ratio=0.2))
        self.next_slide()

        m_mol, m_base = POR_NOME[MELHOR_MOL], POR_NOME[MELHOR_BASE]
        e_mol, e_base = m_mol["fp"] + m_mol["fn"], m_base["fp"] + m_base["fn"]
        if e_mol == e_base:
            detalhe = (f"Mas os dois erram {e_mol} pacientes em 114. Só muda QUAIS: o MOL perdeu {m_mol['fn']} maligno"
                       f"{'s' if m_mol['fn'] != 1 else ''} e a ANN, {m_base['fn']}.")
        else:
            detalhe = f"Mas a diferença é de {abs(e_mol - e_base)} paciente(s) em 114."
        notas = VGroup(
            texto(f"No teste, o melhor MOL aparece na frente ({br(m_mol['teste'], 3)} contra {br(m_base['teste'], 3)}).", 22, COR_TEXTO),
            texto(detalhe, 22, COR_DESTAQUE),
            texto("É o mesmo problema do início: um teste só não separa modelo melhor de modelo com sorte.", 20, COR_SUAVE),
            texto("Por isso a comparação vale pela validação cruzada com o Wilcoxon, não pelo teste isolado.", 20, COR_SUAVE),
        ).arrange(DOWN, buff=0.18).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(notas))
        self.next_slide()
        self.limpar_tudo()

    # =================================================================
    def slide_12_conclusao(self):
        self.trocar_cabecalho("Conclusão da avaliação")
        itens = VGroup(
            texto("1.  Nesta base, o MOL não supera os métodos tradicionais", 26),
            texto("2.  A melhor versão do MOL (ABC) empata estatisticamente com eles", 26),
            texto("3.  As versões que otimizam melhor (PSO e EPSO) são significativamente piores", 26),
            texto("Por quê?  →  análise crítica: o que cada parte do MOL realmente faz", 26, COR_DESTAQUE),
        ).arrange(DOWN, buff=0.5, aligned_edge=LEFT).move_to(DOWN * 0.2)
        itens[3].shift(DOWN * 0.3)
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.3) for i in itens], lag_ratio=0.4), run_time=2.5)
        self.next_slide()
