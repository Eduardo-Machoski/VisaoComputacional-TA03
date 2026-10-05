"""Detecção, ordenação e validação dos quatro cantos do objeto."""

from __future__ import annotations

import math

import cv2
import numpy as np


class CantosNaoEncontrados(ValueError):
    """Nenhum contorno apresentou quatro lados suficientemente definidos."""


def validar_pontos(pontos, forma_imagem: tuple) -> np.ndarray:
    # Identifica os pontos 1-4 conforme a sequencia SE, SD, ID, IE
    # Feito dessa forma pois o objeto pode ter qualquer orientacao na imagem

    p = np.asarray(pontos, dtype=np.float64)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        raise ValueError("Forneça quatro pares (x, y) com valores finitos.")

    altura, largura = forma_imagem[:2]
    if np.any(p < 0) or np.any(p[:, 0] > largura - 1) or np.any(p[:, 1] > altura - 1):
        raise ValueError("Os quatro pontos devem estar dentro da imagem.")

    distancias = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    if np.min(distancias[np.triu_indices(4, 1)]) < 2.0:
        raise ValueError("Há pontos repetidos ou próximos demais entre si.")

    # Garantia que o objeto possui 4 lados e seus cantos foram identificados na ordem correta
    arestas = np.roll(p, -1, axis=0) - p
    seguintes = np.roll(arestas, -1, axis=0)
    cruz = arestas[:, 0] * seguintes[:, 1] - arestas[:, 1] * seguintes[:, 0]
    senos = cruz / (np.linalg.norm(arestas, axis=1) * np.linalg.norm(seguintes, axis=1))

    if np.any(senos <= 1e-3):
        raise ValueError(
            "Selecione um quadrilátero na ordem SE, SD, ID, IE "
            "(horária), sem cruzar lados nem alinhar três pontos."
        )

    return p.astype(np.float32)


# Ordenação ciclica dos cantos do objeto
def ordenar_cantos(pontos) -> np.ndarray:
    p = np.asarray(pontos, dtype=np.float32).reshape(4, 2)
    centro = p.mean(axis=0)
    angulos = np.arctan2(p[:, 1] - centro[1], p[:, 0] - centro[0])
    p = p[np.argsort(angulos)]

    if cv2.contourArea(p, oriented=True) < 0:
        p = p[::-1]

    primeiro = min(
        range(4),
        key=lambda i: (float(p[i].sum()), float(p[i, 1]), float(p[i, 0])),
    )
    return np.roll(p, -primeiro, axis=0).copy()


# Detecta os cantos de um objeto quadrilatero bem definido, utilizando bordas, limiares e contornos.
# Nao considera o fim da imagem como um canto artificial
def detectar_cantos(imagem: np.ndarray, area_minima: float = 0.05) -> np.ndarray:

    h_original, w_original = imagem.shape[:2]
    if min(h_original, w_original) < 32:
        raise CantosNaoEncontrados("Imagem pequena demais para detectar os cantos.")

    escala = min(1.0, 1200 / max(h_original, w_original))
    w, h = round(w_original * escala), round(h_original * escala)
    pequena = cv2.resize(imagem, (w, h), interpolation=cv2.INTER_AREA)
    cinza = cv2.cvtColor(pequena, cv2.COLOR_BGR2GRAY) if pequena.ndim == 3 else pequena

    canais = [cinza]
    if pequena.ndim == 3:
        # Inclui canal de contraste
        componentes = cv2.split(pequena)
        contraste = [
            float(np.percentile(c, 95) - np.percentile(c, 5))
            for c in componentes
        ]
        canais.append(componentes[int(np.argmax(contraste))])

    mascaras = []
    evidencia = np.zeros((h, w), np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    for canal in canais:
        suave = cv2.GaussianBlur(canal, (5, 5), 0)

        for baixo, alto in [(25, 75), (70, 180)]:
            bordas = cv2.Canny(suave, baixo, alto)
            evidencia = cv2.bitwise_or(evidencia, bordas)
            mascaras.append(bordas)
            mascaras.append(
                cv2.morphologyEx(
                    bordas,
                    cv2.MORPH_CLOSE,
                    kernel,
                    iterations=2,
                )
            )

        _, binaria = cv2.threshold(
            suave,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        mascaras.extend([binaria, cv2.bitwise_not(binaria)])

    if not np.any(evidencia):
        raise CantosNaoEncontrados("Não foram encontradas bordas suficientes.")

    # Rejeita contornos muito irregulares e poligonos inventados
    distancia = cv2.distanceTransform(
        cv2.bitwise_not(evidencia),
        cv2.DIST_L2,
        3,
    )

    melhor, melhor_nota = None, -1.0
    area_imagem = float(w * h)

    for mascara in mascaras:
        contornos, _ = cv2.findContours(
            mascara,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contorno in sorted(contornos, key=cv2.contourArea, reverse=True)[:40]:
            area_contorno = cv2.contourArea(contorno)
            if not area_minima * area_imagem <= area_contorno <= 0.98 * area_imagem:
                continue

            perimetro = cv2.arcLength(contorno, True)

            for tolerancia in (0.008, 0.015, 0.025, 0.04):
                poligono = cv2.approxPolyDP(
                    contorno,
                    tolerancia * perimetro,
                    True,
                )

                if len(poligono) != 4 or not cv2.isContourConvex(poligono):
                    continue

                pontos = ordenar_cantos(poligono)

                # Bordas truncadas no limite da foto não definem um objeto completo.
                if (
                    np.any(pontos < 2)
                    or np.any(pontos[:, 0] > w - 3)
                    or np.any(pontos[:, 1] > h - 3)
                ):
                    continue

                try:
                    validar_pontos(pontos, (h, w))
                except ValueError:
                    continue

                area = cv2.contourArea(pontos)
                fidelidade = min(area, area_contorno) / max(area, area_contorno)
                lados = np.roll(pontos, -1, axis=0) - pontos

                if (
                    fidelidade < 0.88
                    or np.min(np.linalg.norm(lados, axis=1)) < 0.06 * min(w, h)
                ):
                    continue

                apoios, distancias = [], []

                for i in range(4):
                    # Ignora as extremidades
                    t = np.linspace(0.05, 0.95, 90)[:, None]
                    amostras = np.rint(pontos[i] + t * lados[i]).astype(int)
                    d = distancia[amostras[:, 1], amostras[:, 0]]
                    apoios.append(float(np.mean(d <= 3.0)))
                    distancias.append(float(np.mean(d)))

                if min(apoios) < 0.70:
                    continue

                # Heuristica para identificar a melhor divisao
                qualidade = 0.65 + 0.25 * np.mean(apoios) + 0.10 * fidelidade
                nota = (
                    (area / area_imagem)
                    * qualidade
                    / (1 + 0.02 * np.mean(distancias))
                )

                if nota > melhor_nota:
                    melhor, melhor_nota = pontos, nota

    if melhor is None:
        raise CantosNaoEncontrados(
            "Não foi encontrado um quadrilátero com quatro lados visíveis. "
        )

    # Reverte o mapeamento de centros de pixel usado por cv2.resize.
    fatores = np.array(
        [w_original / w, h_original / h],
        dtype=np.float32,
    )
    original = (melhor + 0.5) * fatores - 0.5

    return validar_pontos(original, imagem.shape)
