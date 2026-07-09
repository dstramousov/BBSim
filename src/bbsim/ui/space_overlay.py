"""Canvas overlay that visualizes expanding space and current physical scale."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


class SpaceScaleOverlay(QWidget):
    """Transparent overlay with a stretching space grid and scale text."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._lines: tuple[str, ...] = ()
        self._stage_id: str | None = None
        self._stage_progress = 0.0
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setVisible(False)

    def set_overlay_state(
        self,
        *,
        lines: tuple[str, ...],
        stage_id: str | None,
        stage_progress: float,
        visible: bool,
    ) -> None:
        """Update overlay content and redraw."""

        self._lines = lines
        self._stage_id = stage_id
        self._stage_progress = max(0.0, min(1.0, float(stage_progress)))
        self.setVisible(visible)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override name
        """Paint grid and scale readout."""

        _ = event
        if not self.isVisible():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._draw_space_grid(painter)
        if self._lines:
            self._draw_scale_box(painter)

    def _draw_space_grid(self, painter: QPainter) -> None:
        rect = self.rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return

        spacing = self._grid_spacing_px()
        center_x = rect.width() / 2.0
        center_y = rect.height() / 2.0
        inflation_mix = self._inflation_mix()
        alpha = int(26 + 48 * inflation_mix)
        major_alpha = int(42 + 70 * inflation_mix)

        minor_pen = QPen(QColor(140, 190, 255, alpha), 1.0)
        major_pen = QPen(QColor(185, 220, 255, major_alpha), 1.2)

        index = 0
        offset = self._grid_offset_px(spacing)
        x = center_x + offset
        while x <= rect.width() + spacing:
            painter.setPen(major_pen if index % 4 == 0 else minor_pen)
            painter.drawLine(QPointF(x, 0.0), QPointF(x, float(rect.height())))
            if index > 0:
                mirror_x = center_x - (x - center_x)
                painter.drawLine(QPointF(mirror_x, 0.0), QPointF(mirror_x, float(rect.height())))
            x += spacing
            index += 1

        index = 0
        y = center_y + offset
        while y <= rect.height() + spacing:
            painter.setPen(major_pen if index % 4 == 0 else minor_pen)
            painter.drawLine(QPointF(0.0, y), QPointF(float(rect.width()), y))
            if index > 0:
                mirror_y = center_y - (y - center_y)
                painter.drawLine(QPointF(0.0, mirror_y), QPointF(float(rect.width()), mirror_y))
            y += spacing
            index += 1

    def _grid_spacing_px(self) -> float:
        if self._stage_id == "inflation":
            # Inflation should visibly stretch the coordinate net, while the field itself
            # remains a comoving patch. Smoothstep makes the growth feel less mechanical.
            p = self._stage_progress * self._stage_progress * (3.0 - 2.0 * self._stage_progress)
            return 16.0 + 164.0 * p
        if self._stage_id in {"reheating", "nucleosynthesis", "recombination"}:
            return 92.0
        return 58.0

    def _grid_offset_px(self, spacing: float) -> float:
        if self._stage_id != "inflation":
            return 0.0
        # Make the coordinate net breathe outward during inflation instead of only
        # changing spacing. This is a visual cue: space expands everywhere, not from
        # a central explosion.
        p = self._stage_progress * self._stage_progress * (3.0 - 2.0 * self._stage_progress)
        return -0.5 * spacing * p

    def _inflation_mix(self) -> float:
        if self._stage_id != "inflation":
            return 0.0
        return self._stage_progress * self._stage_progress * (3.0 - 2.0 * self._stage_progress)

    def _draw_scale_box(self, painter: QPainter) -> None:
        padding = 10
        line_height = 18
        width = 320
        height = padding * 2 + line_height * len(self._lines)
        box = QRectF(14.0, 14.0, float(width), float(height))

        painter.setPen(QPen(QColor(110, 135, 180, 150), 1.0))
        painter.setBrush(QColor(5, 7, 17, 205))
        painter.drawRoundedRect(box, 8.0, 8.0)

        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

        for index, line in enumerate(self._lines):
            color = QColor(230, 238, 255, 240) if index == 0 else QColor(190, 205, 225, 230)
            painter.setPen(color)
            y = box.top() + padding + line_height * (index + 1) - 4
            painter.drawText(QPointF(box.left() + padding, y), line)
