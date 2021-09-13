from matplotlib import widgets


class CursorCallbackWidget(widgets.AxesWidget):
    """
    Only one callback widget can operate at a time because of the weird background clearing behavior."""

    def __init__(self, ax, move_callbacks, useblit=True, lineprops=None):
        super().__init__(ax)

        self.connect_event("motion_notify_event", self.onmove)
        self.connect_event("draw_event", self.clear)

        self.visible = True
        self.useblit = useblit and self.canvas.supports_blit
        if not self.useblit:
            raise ValueError

        self.move_callbacks = move_callbacks

        if lineprops is None:
            lineprops = [{} for i in range(len(move_callbacks))]

        if self.useblit:
            if lineprops:
                for props in lineprops:
                    props["animated"] = True

        self.plots = [
            ax.plot([0], [0], visible=False, **(lineprops[i]))[0]
            for i in range(len(move_callbacks))
        ]
        self.background = None
        self.needclear = False

    def clear(self, event):
        """Internal event handler to clear the cursor."""
        if self.ignore(event):
            return
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        [plot.set_visible(False) for plot in self.plots]

    def onmove(self, event):
        """Internal event handler to draw the cursor when the mouse moves."""
        if self.ignore(event):
            return
        if not self.canvas.widgetlock.available(self):
            return
        if event.inaxes != self.ax:
            [plot.set_visible(False) for plot in self.plots]

            if self.needclear:
                self.canvas.draw()
                self.needclear = False
            return
        self.needclear = True
        if not self.visible:
            return

        for plot, callback in zip(self.plots, self.move_callbacks):
            x_data, y_data = callback(event.xdata, event.ydata)
            plot.set_xdata(x_data)
            plot.set_ydata(y_data)
            plot.set_visible(self.visible)

        self._update()

    def _update(self):
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            for plot in self.plots:
                self.ax.draw_artist(plot)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()
        return False
