"""Custom Qt timeline progress panel."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from bbsim.core.timeline import DEFAULT_TIMELINE_STAGES, TimelineStage, timeline_progress_position


@dataclass(frozen=True, slots=True)
class TimelineViewState:
    """Current visual state of the pipeline timeline."""

    active_stage_id: str | None = None
    completed_stage_ids: tuple[str, ...] = ()
    local_stage_progress: float = 1.0


class TimelinePanel(QWidget):
    """Draw a full-width visual evolution progress bar with checkpoint labels."""

    def __init__(
        self,
        stages: tuple[TimelineStage, ...] = DEFAULT_TIMELINE_STAGES,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._stages = stages
        self._state = TimelineViewState()
        self.setMinimumHeight(86)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAutoFillBackground(False)

    def set_timeline_state(self, state: TimelineViewState) -> None:
        """Update the timeline state and repaint the widget."""

        self._state = state
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        """Draw the visual timeline."""

        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect().adjusted(8, 4, -8, -6)
        painter.fillRect(rect, QColor(30, 30, 34))

        if not self._stages:
            return

        label_top = rect.top() + 8
        line_y = rect.top() + 46
        left = rect.left() + 34
        right = rect.right() - 34
        width = max(1, right - left)

        points = self._checkpoint_points(left, width, line_y)
        progress = timeline_progress_position(
            self._state.active_stage_id,
            self._state.local_stage_progress,
            self._stages,
        )
        progress_x = left + width * progress

        self._draw_track(painter, left, right, line_y)
        self._draw_filled_track(painter, left, progress_x, line_y)
        self._draw_labels(painter, points, label_top)
        self._draw_checkpoints(painter, points)
        self._draw_current_marker(painter, progress_x, line_y)

    def _checkpoint_points(self, left: int, width: int, line_y: int) -> list[QPointF]:
        if len(self._stages) == 1:
            return [QPointF(left + width / 2, line_y)]
        return [
            QPointF(left + width * index / (len(self._stages) - 1), line_y)
            for index, _stage in enumerate(self._stages)
        ]

    @staticmethod
    def _draw_track(painter: QPainter, left: int, right: int, line_y: int) -> None:
        pen = QPen(QColor(86, 86, 92), 5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(left, line_y), QPointF(right, line_y))

    @staticmethod
    def _draw_filled_track(painter: QPainter, left: int, progress_x: float, line_y: int) -> None:
        if progress_x <= left:
            return
        gradient = QLinearGradient(QPointF(left, line_y), QPointF(progress_x, line_y))
        gradient.setColorAt(0.0, QColor(180, 220, 255))
        gradient.setColorAt(1.0, QColor(255, 238, 174))
        pen = QPen(gradient, 6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(left, line_y), QPointF(progress_x, line_y))

    def _draw_labels(self, painter: QPainter, points: list[QPointF], label_top: int) -> None:
        label_font = QFont(painter.font())
        label_font.setPointSize(max(8, label_font.pointSize() - 1))
        painter.setFont(label_font)

        completed = set(self._state.completed_stage_ids)
        active = self._state.active_stage_id
        for point, stage in zip(points, self._stages, strict=True):
            if stage.stage_id == active:
                color = QColor(245, 245, 245)
            elif stage.stage_id in completed:
                color = QColor(205, 225, 240)
            else:
                color = QColor(132, 132, 138)
            painter.setPen(color)
            label_width = 110
            label_rect = QRectF(point.x() - label_width / 2, label_top, label_width, 18)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, stage.title)

    def _draw_checkpoints(self, painter: QPainter, points: list[QPointF]) -> None:
        completed = set(self._state.completed_stage_ids)
        active = self._state.active_stage_id
        for point, stage in zip(points, self._stages, strict=True):
            if stage.stage_id in completed:
                fill = QColor(230, 242, 255)
                outline = QColor(255, 239, 184)
                radius = 7
            elif stage.stage_id == active:
                fill = QColor(255, 239, 184)
                outline = QColor(255, 255, 255)
                radius = 7
            else:
                fill = QColor(30, 30, 34)
                outline = QColor(125, 125, 132)
                radius = 6

            painter.setPen(QPen(outline, 2))
            painter.setBrush(fill)
            painter.drawEllipse(point, radius, radius)

    @staticmethod
    def _draw_current_marker(painter: QPainter, progress_x: float, line_y: int) -> None:
        marker_path = QPainterPath()
        marker_path.moveTo(progress_x, line_y - 17)
        marker_path.lineTo(progress_x - 8, line_y - 6)
        marker_path.lineTo(progress_x + 8, line_y - 6)
        marker_path.closeSubpath()
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setBrush(QColor(255, 239, 184))
        painter.drawPath(marker_path)
