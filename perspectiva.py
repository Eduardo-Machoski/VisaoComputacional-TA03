from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys


# Fix de warning do qt com relacao a fontes
def _configurar_fontes_qt() -> None:
    if not sys.platform.startswith("linux"):
        return

    import importlib.util

    candidatos = [
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation2"),
        Path("/usr/share/fonts/truetype/freefont"),
    ]

    pasta_fontes = next(
        (pasta for pasta in candidatos if pasta.is_dir() and any(pasta.glob("*.ttf"))),
        None,
    )

    if pasta_fontes is None:
        return

    os.environ.setdefault("QT_QPA_FONTDIR", str(pasta_fontes))

    try:
        spec = importlib.util.find_spec("cv2")
        if spec is None or not spec.submodule_search_locations:
            return

        pasta_cv2 = Path(next(iter(spec.submodule_search_locations)))
        pasta_qt = pasta_cv2 / "qt"
        pasta_destino = pasta_qt / "fonts"

        if pasta_destino.exists() or pasta_destino.is_symlink():
            return

        pasta_qt.mkdir(parents=True, exist_ok=True)
        pasta_destino.symlink_to(pasta_fontes, target_is_directory=True)
    except OSError:
        pass


# Precisa ocorrer antes de importar cv2
_configurar_fontes_qt()

import cv2
import numpy as np

from identificacao_pontos import CantosNaoEncontrados, detectar_cantos, validar_pontos
from transformacao_perspectiva import retificar


# Cancelamento das operacoes
class Cancelado(Exception):
    pass


def ler_imagem(caminho: str | Path) -> np.ndarray:
    caminho = Path(caminho)

    if not caminho.is_file():
        raise ValueError(f"Imagem não encontrada: {caminho}")

    dados = np.fromfile(caminho, dtype=np.uint8)
    if not dados.size:
        raise ValueError("O arquivo da imagem está vazio.")

    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if imagem is None:
        raise ValueError("Não foi possível decodificar a imagem.")

    return imagem


def gravar_imagem(caminho: str | Path, imagem: np.ndarray) -> None:
    caminho = Path(caminho)

    if caminho.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Use uma imagem de saída .png, .jpg ou .jpeg.")

    caminho.parent.mkdir(parents=True, exist_ok=True)
    ok, dados = cv2.imencode(caminho.suffix, imagem)

    if not ok:
        raise ValueError("Falha ao codificar a imagem de saída.")

    dados.tofile(caminho)


# Redimensiona a tela, imagem original se mantem igual
def ajustar_visualizacao(imagem: np.ndarray, max_w=1280, max_h=920):
    h, w = imagem.shape[:2]
    fator = min(1.0, max_w / w, max_h / h)
    dw, dh = max(2, round(w * fator)), max(2, round(h * fator))

    # Redimensiona a imagem caso necessario
    miniatura = cv2.resize(
        imagem,
        (dw, dh),
        interpolation=cv2.INTER_AREA,
    )

    escala = np.array(
        [(w - 1) / (dw - 1), (h - 1) / (dh - 1)],
        dtype=float,
    )

    return miniatura, escala


def exigir_interface() -> None:
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise ValueError(
            "Sem tela gráfica disponível para selecionar e confirmar os pontos."
        )

    gui = [
        linha.strip()
        for linha in cv2.getBuildInformation().splitlines()
        if "GUI:" in linha
    ]

    if gui and ("NONE" in gui[0] or gui[0].endswith("NO")):
        raise ValueError(
            "Este OpenCV não tem interface gráfica. Instale opencv-python."
        )


# Teça para selecionar e ajustar os pontos do documento
def selecionar_pontos(imagem: np.ndarray, iniciais=None) -> np.ndarray:
    exigir_interface()

    miniatura, escala = ajustar_visualizacao(imagem)
    pontos = [] if iniciais is None else [list(map(float, p)) for p in iniciais]

    topo = 10
    nome = "Plano Frontal | selecao"
    mensagem = (
        "Cantos detectados. Enter: confirmar | arraste para ajustar."
        if iniciais is not None
        else "Clique: 1 SE, 2 SD, 3 ID, 4 IE (ordem horaria do objeto)."
    )
    arrastando = None

    # Identifica qual ponto esta sendo movido e salva sua nova posicao
    def mouse(evento, x, y, flags, userdata):
        nonlocal mensagem, arrastando

        if evento == cv2.EVENT_LBUTTONUP:
            arrastando = None
            return

        dentro = (
            0 <= x < miniatura.shape[1]
            and topo <= y < topo + miniatura.shape[0]
        )

        if evento == cv2.EVENT_MOUSEMOVE and arrastando is not None and dentro:
            pontos[arrastando] = [
                x * escala[0],
                (y - topo) * escala[1],
            ]
            return

        if evento == cv2.EVENT_RBUTTONDOWN and pontos and arrastando is None:
            pontos.pop()

        if evento == cv2.EVENT_LBUTTONDOWN and dentro:
            if pontos:
                dist = np.linalg.norm(
                    np.asarray(pontos) / escala - [x, y - topo],
                    axis=1,
                )

                if dist.min() <= 18:
                    arrastando = int(dist.argmin())
                    mensagem = "Ajustando canto. Enter: confirmar | R: limpar."
                    return

            if len(pontos) < 4:
                pontos.append(
                    [
                        x * escala[0],
                        (y - topo) * escala[1],
                    ]
                )
                mensagem = f"{len(pontos)}/4 pontos selecionados."

    cv2.namedWindow(nome, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(nome, mouse)

    try:
        while True:
            painel = np.full(
                (
                    miniatura.shape[0] + topo,
                    max(760, miniatura.shape[1]),
                    3,
                ),
                27,
                np.uint8,
            )
            painel[topo:, : miniatura.shape[1]] = miniatura

            linhas = [
                "1 SE > 2 SD > 3 ID > 4 IE | Enter: confirmar",
                "R: limpar | Backspace / botao direito: desfazer | Esc: cancelar",
                mensagem,
            ]

            for i, texto in enumerate(linhas):
                cv2.putText(
                    painel,
                    texto,
                    (12, 24 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )

            exibidos = [
                np.rint(np.array(p) / escala + [0, topo]).astype(int)
                for p in pontos
            ]

            if len(exibidos) > 1:
                cv2.polylines(
                    painel,
                    [np.array(exibidos, dtype=np.int32)],
                    len(exibidos) == 4,
                    (30, 220, 150),
                    2,
                    cv2.LINE_AA,
                )

            for i, p in enumerate(exibidos):
                cv2.circle(
                    painel,
                    tuple(p),
                    6,
                    (30, 220, 150),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    painel,
                    str(i + 1),
                    tuple(p + [9, -9]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    painel,
                    str(i + 1),
                    tuple(p + [9, -9]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(nome, painel)
            tecla = cv2.waitKey(30) & 0xFF

            if tecla == 27 or cv2.getWindowProperty(nome, cv2.WND_PROP_VISIBLE) < 1:
                raise Cancelado()

            if tecla in (ord("r"), ord("R")):
                pontos.clear()
                arrastando = None
                mensagem = "Selecao limpa. Clique nos quatro cantos."

            if tecla in (8, 127) and pontos:
                pontos.pop()
                arrastando = None

            if tecla in (10, 13):
                try:
                    return validar_pontos(pontos, imagem.shape)
                except ValueError:
                    mensagem = (
                        "Selecao invalida. Use 4 cantos em ordem horaria; "
                        "R para refazer."
                    )
    finally:
        cv2.destroyWindow(nome)


# Mostra a imagem transformada a partir dos pontos
# Permite retornar a original para ajustar, salvar a imagem ou cancelar
def confirmar_resultado(resultado: np.ndarray) -> bool:
    exigir_interface()

    miniatura, _ = ajustar_visualizacao(resultado)
    painel = np.full(
        (
            miniatura.shape[0] + 55,
            max(650, miniatura.shape[1]),
            3,
        ),
        27,
        np.uint8,
    )
    painel[55:, : miniatura.shape[1]] = miniatura

    cv2.putText(
        painel,
        "S: salvar | R: refazer pontos | Esc: cancelar",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    nome = "Plano Frontal | resultado"
    cv2.namedWindow(nome, cv2.WINDOW_AUTOSIZE)

    try:
        cv2.imshow(nome, painel)

        while True:
            tecla = cv2.waitKey(30) & 0xFF

            if tecla == 27 or cv2.getWindowProperty(nome, cv2.WND_PROP_VISIBLE) < 1:
                raise Cancelado()

            if tecla in (ord("s"), ord("S")):
                return True

            if tecla in (ord("r"), ord("R")):
                return False
    finally:
        cv2.destroyWindow(nome)




# Interface da linha de comando
def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "imagem",
        type=Path,
        help="Fotografia do quadro, cartaz ou documento",
    )

    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("saida/retificado.png"),
    )

    return parser


def main(argv=None) -> int:
    # Obtem argumentos da linha de comando
    parser = criar_parser()
    args = parser.parse_args(argv)

    try:
        entrada = args.imagem.resolve()
        saida = args.saida.resolve()

        if saida.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("Use uma saída .png, .jpg ou .jpeg.")

        # Obtem imagem a ser transformada
        imagem = ler_imagem(entrada)
        pontos = None

        # Selecao dos pontos que definem os 4 cantos do objeto
        try:
            pontos = detectar_cantos(imagem)
            print("Quatro cantos detectados automaticamente.")
        except CantosNaoEncontrados as erro:
            print(f"{erro}\nAbrindo seleção manual.")

        # Ajuste manual dos cantos, para melhorar acuracia
        if pontos is not None:
            pontos = selecionar_pontos(imagem, iniciais=pontos)
        while True:
            if pontos is None:
                pontos = selecionar_pontos(imagem)

            # Transformacao do objeto (pode ser retornada, por isso esta dentro do loop)
            resultado = retificar(
                imagem,
                pontos
            )

            if confirmar_resultado(resultado):
                break

            pontos = None

        gravar_imagem(saida, resultado)
        print(f"Imagem transformada salva em: {args.saida}")

        return 0

    except Cancelado:
        print("Operação cancelada; nenhuma saída foi gravada.")
        return 0

    except (ValueError, OSError, cv2.error, TypeError) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
