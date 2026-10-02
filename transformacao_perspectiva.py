"""Cálculo da homografia e transformação perspectiva da imagem."""

from __future__ import annotations

import math

import cv2
import numpy as np

from identificacao_pontos import validar_pontos

# Estima as dimensoes do objeto retangular transformado
# Considera distorcao de perspectiva e assume:
#  - Lados opostos paralelos no objeto real
#  - Lados adjacentes perpendiculares
def dimensoes_saida(
    pontos: np.ndarray,
    forma_imagem: tuple,
) -> tuple[int, int]:

    pontos = np.asarray(pontos, dtype=np.float64)

    se, sd, id_, ie = pontos

    altura_img, largura_img = forma_imagem[:2]

    cx = (largura_img - 1) / 2.0
    cy = (altura_img - 1) / 2.0

    def ponto_homogeneo(p):
        return np.array([p[0], p[1], 1.0], dtype=np.float64)

    def reta(p1, p2):
        return np.cross(
            ponto_homogeneo(p1),
            ponto_homogeneo(p2),
        )

    # Pontos de fuga
    reta_superior = reta(se, sd)
    reta_inferior = reta(ie, id_)

    fuga_horizontal = np.cross(
        reta_superior,
        reta_inferior,
    )

    # Direção vertical
    reta_esquerda = reta(se, ie)
    reta_direita = reta(sd, id_)

    fuga_vertical = np.cross(
        reta_esquerda,
        reta_direita,
    )

    # Estimativa da distancia focal -> fuga_horizontal e fuga_vertical sao perpendiculares
    vh = fuga_horizontal
    vv = fuga_vertical

    usar_estimativa_perspectiva = (
        abs(vh[2]) > 1e-8
        and abs(vv[2]) > 1e-8
    )

    proporcao = None

    if usar_estimativa_perspectiva:

        # Produto escalar entre: vetores que saem do centro da imagem ate cada ponto de fuga
        numerador = (
            (vh[0] - cx * vh[2])
            * (vv[0] - cx * vv[2])
            +
            (vh[1] - cy * vh[2])
            * (vv[1] - cy * vv[2])
        )

        # Normalizacao das coordenadas
        denominador = vh[2] * vv[2]

        f2 = -numerador / denominador

        if np.isfinite(f2) and f2 > 0:

            f = np.sqrt(f2)

            origem_unitaria = np.float32(
                [
                    [0, 0],
                    [1, 0],
                    [1, 1],
                    [0, 1]
                ]
            )

            destino = pontos.astype(np.float32)

            H = cv2.getPerspectiveTransform(
                origem_unitaria,
                destino
            )

            # Matriz intrínseca aproximada
            K_inv = np.array(
                [
                    [1.0 / f, 0, -cx / f],
                    [0, 1.0 / f, -cy / f],
                    [0, 0, 1]
                ],
                dtype=np.float64,
            )

            # As duas primeiras colunas representam
            # as direções horizontal e vertical do plano.
            vetor_horizontal = K_inv @ H[:, 0]
            vetor_vertical = K_inv @ H[:, 1]

            largura_real = np.linalg.norm(vetor_horizontal)
            altura_real = np.linalg.norm(vetor_vertical)


    # Caso pontos de fuga forem muito distantes, ou estimativa nao for estavel
    largura_superior = np.linalg.norm(sd - se)
    largura_inferior = np.linalg.norm(id_ - ie)

    altura_esquerda = np.linalg.norm(ie - se)
    altura_direita = np.linalg.norm(id_ - sd)

    if proporcao is None or not np.isfinite(proporcao) or proporcao <= 0:

        largura_media = (
            largura_superior + largura_inferior
        ) / 2.0

        altura_media = (
            altura_esquerda + altura_direita
        ) / 2.0

        proporcao = largura_media / altura_media

    # Define a resolucao da imagem de saida
    w = max(
        2,
        round(max(largura_superior, largura_inferior)),
    )

    h = max(
        2,
        round(w / proporcao),
    )

    return w, h


def retificar(
    imagem: np.ndarray,
    pontos,
) -> np.ndarray:

    if imagem is None or imagem.size == 0 or imagem.ndim not in (2, 3):
        raise ValueError("A imagem de entrada está vazia ou é inválida.")

    # Obtem os pontos dos cantos e a dimensao aproximada do objeto
    origem = validar_pontos(pontos, imagem.shape)
    w, h = dimensoes_saida(origem, imagem.shape)

    # Define os cantos da imagem final
    destino = np.float32(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ]
    )

    # Transforma a imagem
    matriz = cv2.getPerspectiveTransform(origem, destino)
    resultado = cv2.warpPerspective(
        imagem,
        matriz,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return resultado
