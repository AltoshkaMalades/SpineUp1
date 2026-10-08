"""Short desk-friendly posture exercises for the 1-minute break."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Exercise:
    name: str
    description: str
    seconds: int
    animation: str


EXERCISES: tuple[Exercise, ...] = (
    Exercise(
        "Chin tuck (втягивание подбородка)",
        "Сядьте прямо и медленно отведите подбородок назад, как будто делаете "
        "двойной подбородок. Держите 3 секунды, расслабьтесь, повторяйте.",
        20,
        "assets/впк.gif"
    ),
    Exercise(
        "Раскрытие грудной клетки",
        "Соедините руки за спиной, разведите плечи назад и слегка поднимите "
        "грудь. Дышите ровно, не прогибайте поясницу.",
        20,
        "assets/мек.gif"
    ),
    Exercise(
        "Повороты корпуса сидя",
        "Положите правую руку на левое колено и мягко повернитесь влево, "
        "глядя через плечо. Через 10 секунд смените сторону.",
        20,"assets/пкк.gif"
    ),
)
