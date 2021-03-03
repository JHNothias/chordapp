print('Loading ui libraries...')
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
import pyqtgraph as pg
import qdarkstyle
import sys
import conveniently
print('Loading functional libraries...')
import ProcessAudio as pa
import pydub as pd
import numpy as np
print('imports OK.')


# Convenience Functions, classes and definitions ---
def anEmptyClass():
    class Empty: pass
    return Empty

GreenPen = pg.mkPen(color = "#BFFF00", width = 1)
BluePen = pg.mkPen(color = "#00AAFF", width = 1)
OrangePen = pg.mkPen(color = "#FFAA00", width = 1)

def buildVLine():
    VLine = QFrame()
    VLine.setFrameShape(QFrame.VLine)
    VLine.setFrameShadow(QFrame.Sunken)
    return VLine

def buildHLine():
    HLine = QFrame()
    HLine.setFrameShape(QFrame.HLine)
    HLine.setFrameShadow(QFrame.Sunken)
    return HLine

computeHCenters = conveniently.delegated(pa.computeHCenters)
computeVariance = conveniently.delegated(pa.computeVariance)
computeH = conveniently.delegated(pa.computeH)
computeCoh = conveniently.delegated(pa.computeCoh)

toPlaybackPos = lambda Self, Pos, DataLength: int(Self.Player.PlaybackSlider.maximum()*Pos/DataLength)

def centerViewOn(ViewBox, X):
    Radius = ViewBox.viewRect().width()/2
    ViewBox.setXRange(X-Radius, X+Radius, padding = 0)

# -----------------------


# Processing ---
def loadinPlayer(Namespace, Path):
    print('setting up audio playback...')
    try:
        Namespace.MediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(Path)))
        if Namespace.PlayButton.isChecked():
            Namespace.PlayButton.toggle()
        print('done.')
    except:
        print('Error: the selected file could not be read.')


@conveniently.threaded
def loadinAudioProcessing(Self):
    print('loading audio for processing...')

    path = Self.CurrentFilePath
    if path.lower().endswith('.mp3'):
        Audio = pd.AudioSegment.from_file(path, format="mp3")
    elif path.lower().endswith('.wav'):
        Audio = pd.AudioSegment.from_file(path, format="wav")
    elif path.lower().endswith('.raw'):
        Audio = pd.AudioSegment.from_file(path, format="raw")
    elif path.lower().endswith('.ogg'):
        Audio = pd.AudioSegment.from_file(path, format="raw")
    elif path.lower().endswith('.flv'):
        Audio = pd.AudioSegment.from_file(path, format="flv")
    else:
        print('Loading Failed.')
        return

    P = Self.Processing
    P.AudioSegment = Audio.set_channels(1)
    P.sr = Audio.frame_rate
    print('converting audio...')
    P.Audio = np.array(P.AudioSegment.get_array_of_samples()).astype(np.int64)/(2**(Audio.sample_width*8))
    print('drawing audio...')
    Self.Plots.doDrawAudio = True
    updatePlot(Self, Self.Plots.Audio)
    print('calculating chromagram...')
    P.Chromagram = pa.getChromagram(P.Audio, P.sr)
    P.AudioSampleLength = P.Audio.shape[0]
    P.AudioDuration = Audio.duration_seconds

    print('calculating features...')
    updateProcessedChromaData(Self)
    updateRawFeatures(Self)
    print('updating plots...')
    updatePlots(Self)
    print('done.')


@conveniently.threaded
# Modify if you want to add new data to plot
def updatePlot(Self, P):
    if P.isActive:
        if P.Name == 'Audio' and Self.Plots.doDrawAudio:
            if P.PostSmoothingW > 0:
                P.Data = [pa.smoothout(Self.Processing.Audio, P.PostSmoothingW)]
            else:
                P.Data = [Self.Processing.Audio]
            P.DataLength = np.shape(P.Data[0])[0]
            P.Lines[0].setData(P.Data[0])
            Self.Plots.doDrawAudio = False
        elif P.Name == 'Harmonic Center':
            if P.PostSmoothingW > 0:
                P.Data = [pa.smoothout(Self.Processing.RawC, P.PostSmoothingW)]
            else:
                P.Data = [Self.Processing.RawC]
            P.DataLength = np.shape(P.Data[0])[0]
            P.Lines[0].setData(P.Data[0])
        elif P.Name == 'Variance':
            if P.PostSmoothingW > 0:
                P.Data = [pa.smoothout(Self.Processing.RawV, P.PostSmoothingW)]
            else:
                P.Data = [Self.Processing.RawV]
            P.DataLength = np.shape(P.Data[0])[0]
            P.Lines[0].setData(P.Data[0])
        elif P.Name == 'Harmoniousness':
            if P.PostSmoothingW > 0:
                P.Data = [pa.smoothout(Self.Processing.RawH, P.PostSmoothingW)]
            else:
                P.Data = [Self.Processing.RawH]
            P.DataLength = np.shape(P.Data[0])[0]
            P.Lines[0].setData(P.Data[0])
        elif P.Name == 'Coharmoniousness':
            if P.PostSmoothingW > 0:
                P.Data = [pa.smoothout(Self.Processing.RawCoh, P.PostSmoothingW)]
            else:
                P.Data = [Self.Processing.RawCoh]
            P.DataLength = np.shape(P.Data[0])[0]
            P.Lines[0].setData(P.Data[0])
        elif P.Name == 'Both H and CoH':
            if P.PostSmoothingW > 0:
                P.Data = [pa.smoothout(Self.Processing.RawCoh, P.PostSmoothingW), pa.smoothout(Self.Processing.RawH, P.PostSmoothingW)]
            else:
                P.Data = [Self.Processing.RawCoh, Self.Processing.RawH]
            P.DataLength = max(np.shape(P.Data[0])[0], np.shape(P.Data[1])[0])
            P.Lines[0].setData(P.Data[0])
            P.Lines[1].setData(P.Data[1])
        elif P.Name == 'H times CoH':
            d = Self.Processing.RawCoh * Self.Processing.RawH
            s = 2*(d>0)-1
            if P.PostSmoothingW > 0:
                #s*np.sqrt(np.abs(d))
                P.Data = [pa.smoothout(s*np.sqrt(np.abs(d)), P.PostSmoothingW)]
            else:
                P.Data = [s*np.sqrt(np.abs(d))]
            P.DataLength = np.shape(P.Data[0])[0]
            P.Lines[0].setData(P.Data[0])


def updatePlots(Self):
    for P in Self.Plots.All:
        updatePlot(Self, P)


def updateProcessedChromaData(Self):
    print('updating chroma data...')
    D, C = pa.quickprocessChromagram(Self.Processing.Chromagram, treshold=0)
    Self.Processing.Distributions = D
    Self.Processing.Centroids = C


# Modify if you want to add more complex feature data
def updateRawFeatures(Self):
    print('updating Feature data...')
    Cp = computeHCenters(Self.Processing.Centroids)
    Vp = computeVariance(Self.Processing.Centroids)
    Hp = computeH(Self.Processing.Distributions, Self.Processing.Centroids)
    CHp = computeCoh(Self.Processing.Distributions, Self.Processing.Centroids)
    Self.Processing.RawC = Cp.result()
    Self.Processing.RawV = Vp.result()
    Self.Processing.RawH = Hp.result()
    Self.Processing.RawCoh = CHp.result()
    Self.Processing.RawFeatureLength = Self.Processing.RawC.shape[0]
# -----------------------


# Building ---
def buildPlaybackTimer(Self, Namespace):
    Timer = QTimer()
    Timer.setTimerType(Qt.PreciseTimer)
    Timer.setInterval(50)

    def _onTimeout():
        if not Namespace.TimeCheck and Namespace.PlaybackSlider.value() < Namespace.Duration:
            Namespace.PlaybackSlider.setValue(Namespace.PlaybackSlider.value()+50)
            try:
                for B, L, V, T in [(P.PlaybackBar, P.DataLength, P.ViewBox, P.isTracking) for P in Self.Plots.All if P.isActive]:
                    Pos = L*Self.Player.Progress
                    B.setValue(Pos)
                    if T:
                        centerViewOn(V, Pos)
            except:
                print("Error: Timer reposition failed.")
                pass
    Timer.timeout.connect(_onTimeout)

    Namespace.PlaybackTimer = Timer
    return Timer


def buildMediaPlayer(Self, Namespace):
    Namespace.MediaPlayer = QMediaPlayer()

    Namespace.TimeCheck = False

    _onEndOfMedia = lambda : Namespace.PlayButton.toggle() if Namespace.MediaPlayer.mediaStatus() == 7 else None

    def _onPositionChanged(Position):
        Namespace.TimeCheck = True
        Namespace.PlaybackSlider.setValue(Position)
        Namespace.TimeCheck = False

    def _onDurationChanged(Duration):
        Namespace.Duration = Duration
        Namespace.PlaybackSlider.setRange(0, Duration)

    def _onStateChanged():
        if Namespace.MediaPlayer.state()%2:
            Namespace.PlaybackTimer.start()
        else:
            Namespace.PlaybackTimer.stop()

    Namespace.MediaPlayer.positionChanged.connect(_onPositionChanged)
    Namespace.MediaPlayer.durationChanged.connect(_onDurationChanged)
    Namespace.MediaPlayer.mediaStatusChanged.connect(_onEndOfMedia)
    Namespace.MediaPlayer.stateChanged.connect(_onStateChanged)

    Namespace.MediaPlayer.setVolume(80)


def buildPlayButton(Self, Namespace):
    Button = QPushButton()
    Button.setText('Play')
    Button.setCheckable(True)
    Button.setMinimumWidth(50)
    Button.setMaximumWidth(50)
    def _onclicked():
        if Button.isChecked():
            Namespace.MediaPlayer.play()
        else:
            Namespace.MediaPlayer.pause()
    Button.clicked.connect(_onclicked)

    Namespace.PlayButton = Button

    return Button


def buildVolumeSlider(Self, Namespace):
    Layout = QHBoxLayout()
    Name = QLabel('Vol ')
    Slider = QSlider(Qt.Horizontal)

    Slider.setMinimumWidth(160)
    Slider.setMaximumWidth(160)
    Slider.setMinimum(0)
    Slider.setMaximum(100)
    Slider.setValue(80)

    def _onSliderInterraction():
        Namespace.MediaPlayer.setVolume(Slider.value())
    Slider.sliderMoved.connect(_onSliderInterraction)
    Slider.sliderReleased.connect(_onSliderInterraction)

    Layout.addWidget(Name)
    Layout.addWidget(Slider)
    Namespace.VolumeSlider = Slider
    Namespace.VolumeLabel = Name
    Namespace.VolumeSliderLayout = Layout
    return Layout


def buildPlaybackSlider(Self, Namespace):
    Slider = QSlider(Qt.Horizontal)

    def _onValueChanged():
        Namespace.Progress = Slider.value()/Slider.maximum()
    def _onSliderInterraction():
        Namespace.MediaPlayer.setPosition(Slider.value())
        for B, L, V, T in [(P.PlaybackBar, P.DataLength, P.ViewBox, P.isTracking) for P in Self.Plots.All if P.isActive]:
            Pos = L*Namespace.Progress
            B.setValue(Pos)
            if T:
                centerViewOn(V, Pos)
    Slider.sliderMoved.connect(_onSliderInterraction)
    Slider.sliderReleased.connect(_onSliderInterraction)
    Slider.valueChanged.connect(_onValueChanged)
    Namespace.PlaybackSlider = Slider
    return Slider


def buildOpenButton(Self):
    Button = QPushButton()
    Button.setText('Load')
    Button.setMinimumWidth(50)
    Button.setMaximumWidth(50)

    FileLoader = QFileDialog()
    FileLoader.setAcceptMode(QFileDialog.AcceptOpen)
    FileLoader.setNameFilters(['*.wav', '*.mp3', '*.ogg', '*.raw', '.flv'])

    def _runFileLoader():
        print('running file loader.')
        try:
            FileLoader.exec_()
            Self.CurrentFilePath = FileLoader.selectedFiles()[0]
            Self.Player.CurrentFilePath = Self.CurrentFilePath
        except:
            print("Error: file loader failed.")
            return
        loadinPlayer(Self.Player, Self.CurrentFilePath)
        loadinAudioProcessing(Self)

    Button.clicked.connect(_runFileLoader)

    return Button


def buildPlotDisplayArea(Self):
    """
    builds the widget which houses the plots.
    """
    ScrollArea = QScrollArea()
    MainWidget = QWidget()
    Self.PlotArea.Layout = QVBoxLayout()
    ScrollArea.setWidgetResizable(True)
    MainWidget.setLayout(Self.PlotArea.Layout)
    ScrollArea.setWidget(MainWidget)
    return ScrollArea


def buildPlot(Self, Namespace):
    """
    creates the plot widget as well as controls for layout interractions.
    """
    # Widgets and stuff ---
    Support = QWidget()
    Layout = QHBoxLayout()
    PlotLayout = QVBoxLayout()
    NameLabel = QLabel(Namespace.Name)
    NameLabel.setStyleSheet('QLabel {color : #FFAA00}')
    # -----------------------


    # Plot Setup ---
    PlotW = pg.PlotWidget()
    PlotW.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
    PlotI = PlotW.getPlotItem()
    PlotI.setDownsampling(True, True, 'peak')
    PlotI.showGrid(True, True, .5)
    PlotI.setClipToView(True)


    Namespace.Lines = [pg.PlotDataItem([], pen=BluePen)]
    PlotI.addItem(Namespace.Lines[0])
    if Namespace.Name == 'Both H and CoH': # when multiple lines are displayed
        Namespace.Lines.append(pg.PlotDataItem([], pen=OrangePen))
        PlotI.addItem(Namespace.Lines[1])

    Namespace.ViewBox = PlotI.getViewBox()
    Namespace.PlotItem = PlotI
    # -----------------------


    # Layout Interactivity ---
    PlaybackBar = pg.InfiniteLine(0, pen = GreenPen, movable = True)
    def _onSigDragged():
        Pos = toPlaybackPos(Self, PlaybackBar.value(), Namespace.DataLength)
        Self.Player.PlaybackSlider.setValue(Pos)
        Self.Player.MediaPlayer.setPosition(Pos)
    PlaybackBar.sigDragged.connect(_onSigDragged)
    PlotI.addItem(PlaybackBar)
    Namespace.PlaybackBar = PlaybackBar


    MoveUpButton = QPushButton()
    MoveUpButton.setText('/\\')
    MoveUpButton.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
    def _onClickedMoveUpButton():
        Index = Self.PlotArea.Layout.indexOf(Support)
        if Index in {0, -1}:
            return
        OtherWidget = Self.PlotArea.Layout.itemAt(Index -1).widget()
        Self.PlotArea.Layout.removeWidget(OtherWidget)
        Self.PlotArea.Layout.removeWidget(Support)
        Self.PlotArea.Layout.insertWidget(Index -1, Support)
        Self.PlotArea.Layout.insertWidget(Index, OtherWidget)
    MoveUpButton.clicked.connect(_onClickedMoveUpButton)


    SmoothingWidget = QWidget()
    SmoothingWidget.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
    SmoothingLabel = QLabel('Smoothing (samples)')
    SmoothingEdt = QLineEdit()
    def _onEditingFinished():
        try:
            Namespace.PostSmoothingW = int(SmoothingEdt.text())
        except:
            Namespace.PostSmoothingW = 0
        if Namespace.Name == 'Audio':
            Self.Plots.doDrawAudio = True
        updatePlot(Self, Namespace)
    SmoothingEdt.editingFinished.connect(_onEditingFinished)
    SmoothingLayout = QVBoxLayout()
    SmoothingLayout.addWidget(SmoothingLabel)
    SmoothingLayout.addWidget(SmoothingEdt)
    SmoothingWidget.setLayout(SmoothingLayout)


    TrackingToggleButton = QPushButton()
    TrackingToggleButton.setCheckable(True)
    TrackingToggleButton.setText('Follow')
    TrackingToggleButton.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
    def _onClickedTrackingToggleButton():
        Namespace.isTracking = TrackingToggleButton.isChecked()
    TrackingToggleButton.clicked.connect(_onClickedTrackingToggleButton)


    MoveDownButton = QPushButton()
    MoveDownButton.setText('\\/')
    MoveDownButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
    def _onClickedMoveDownButton():
        Index = Self.PlotArea.Layout.indexOf(Support)
        if Index in {Self.PlotArea.Layout.count()-1, -1}:
            return
        OtherWidget = Self.PlotArea.Layout.itemAt(Index +1).widget()
        Self.PlotArea.Layout.removeWidget(OtherWidget)
        Self.PlotArea.Layout.removeWidget(Support)
        Self.PlotArea.Layout.insertWidget(Index, OtherWidget)
        Self.PlotArea.Layout.insertWidget(Index +1, Support)
    MoveDownButton.clicked.connect(_onClickedMoveDownButton)


    ControlsLayout = QVBoxLayout()
    ControlsLayout.addWidget(MoveUpButton)
    ControlsLayout.addWidget(SmoothingWidget)
    ControlsLayout.addWidget(TrackingToggleButton)
    ControlsLayout.addWidget(MoveDownButton)
    # -----------------------


    PlotLayout.addWidget(NameLabel)
    PlotLayout.addWidget(PlotW)
    Layout.addLayout(PlotLayout)
    Layout.addLayout(ControlsLayout)
    Support.setLayout(Layout)
    Self.PlotArea.Layout.addWidget(Support)
    return Support


def buildPlotButton(Self, Namespace):
    """
    builds the button which triggers a plot to show up.
    """
    Button = QPushButton()
    Button.setCheckable(True)
    Button.setText(Namespace.Name)
    Button.setFixedSize(200, 25)
    Button.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))

    def _onClicked():
        Namespace.isActive = Button.isChecked()
        if Button.isChecked():
            Namespace.Widget = buildPlot(Self, Namespace)
            if Namespace.Name == 'Audio':
                Self.Plots.doDrawAudio = True
            updatePlot(Self, Namespace)
        else:
            Self.PlotArea.Layout.removeWidget(Namespace.Widget)
            Namespace.Widget.close()

    Button.clicked.connect(_onClicked)
    return Button


def buildMainWindow(Self):
    """
    creates the window widget, and places layouts and widgets.
    """
    # Widgets ---
    Window = QWidget()
    buildMediaPlayer(Self, Self.Player)
    buildPlaybackTimer(Self, Self.Player)
    # -----------------------


    # Layout ---
    MainLayout = QHBoxLayout()
    LeftLayout = QVBoxLayout()

    LeftTopContainer = QWidget()
    LeftTopContainer.setFixedSize(200, 150)
    LeftTopContainerLayout =QVBoxLayout()
    LeftTopLayout = QHBoxLayout()
    LeftTopLayout.addWidget(buildOpenButton(Self))
    LeftTopLayout.addWidget(buildPlayButton(Self, Self.Player))
    LeftTopContainerLayout.addLayout(LeftTopLayout)
    LeftTopContainerLayout.addLayout(buildVolumeSlider(Self, Self.Player))
    LeftTopContainer.setLayout(LeftTopContainerLayout)
    LeftLayout.addWidget(LeftTopContainer)

    LeftBottomLayout = QVBoxLayout()
    for Namespace in Self.Plots.All:
        LeftBottomLayout.addWidget(buildPlotButton(Self, Namespace))
    LeftLayout.addLayout(LeftBottomLayout)

    MainLayout.addLayout(LeftLayout)
    MainLayout.addWidget(buildVLine())

    RightLayout = QVBoxLayout()
    RightLayout.addWidget(buildPlaybackSlider(Self, Self.Player))
    RightLayout.addWidget(buildPlotDisplayArea(Self))
    MainLayout.addLayout(RightLayout)
    # -----------------------


    Window.setLayout(MainLayout)
    return Window
# -----------------------

# Make sure namespaces are coherent if you add anything.
def Application():
    """
    Initiates Application, Namespaces and spawns the main window.
    Namespaces are as follows:
    Self.
    |   CurrentFilePath:    str
    |   Player.
    |   |   MediaPlayer:        QMediaPlayer
    |   |   PlaybackTimer:      QTimer
    |   |   PlaybackSlider:     QSlider
    |   |   VolumeSlider:       QSlider
    |   |   VolumeLabel:        QLabel
    |   |   VolumeSliderLayout: QHBoxLayout
    |   |   PlayButton:         QPushButton
    |   |   TimeCheck:          Bool
    |   |   Progress:           float
    #           do PlaybackSlider.value()/PlaybackSlider.maximum()
    |   Processing.
    |   |   AudioSegment:       pd.AudioSegment
    |   |   sr:                 int
    |   |   Audio:              np.ndArray
    |   |   Chromagram:         np.ndArray
    |   |   AudioSampleLength:  int
    |   |   AudioDuration:      float
    |   |   Distributions:      np.ndArray
    |   |   Centroids:          np.ndArray
    |   |   RawFeatureLength:
    |   |   RawC
    |   |   RawV
    |   |   RawH
    |   |   RawCoh
    |   Plots.
    |   |   All:            list of all namespaces (classes).
    |   |   doDrawAudio:    bool
    |   |   Audio:          class
    |   |   C:              class
    |   |   V:              class
    |   |   H:              class
    |   |   CH:             class
    |   |   HCH:            class
            HxCH:           class
    |   |   <Namespace>
    |   |   |   Name:           str
    |   |   |   isActive:       bool, if the plot is visible
    |   |   |   Line:           pg.PlotDataItem
    |   |   |   ViewBox:        pg.ViewBox
    |   |   |   isTracking:     bool
    |   |   |   PlaybackBar:    pg.InfiniteLine
    |   |   |   DataLength:     int
    |   |   |   Data:           np.ndArray
    |   PlotArea.
    |   |   Layout: QVBoxLayout
    """
    print('initialisation...')
    App = QApplication(sys.argv)
    App.setApplicationName("Harmonic quality")
    App.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    # Setup Namespaces ---
    Self = anEmptyClass()
    Self.Player = anEmptyClass()
    Self.Processing = anEmptyClass()
    Self.Plots = anEmptyClass()
    Self.Plots.doDrawAudio = False

    Self.Plots.Audio = anEmptyClass()
    Self.Plots.C = anEmptyClass()
    Self.Plots.V = anEmptyClass()
    Self.Plots.H = anEmptyClass()
    Self.Plots.CH = anEmptyClass()
    Self.Plots.HCH = anEmptyClass()
    Self.Plots.HxCH = anEmptyClass()

    Self.PlotArea = anEmptyClass()
    Self.Plots.All = [Self.Plots.C, Self.Plots.H, Self.Plots.CH, Self.Plots.HCH, Self.Plots.Audio, Self.Plots.V, Self.Plots.HxCH]
    for Plot in Self.Plots.All: Plot.isActive = False
    for Plot in Self.Plots.All: Plot.isTracking = False
    for Plot in Self.Plots.All: Plot.PostSmoothingW = 0
    Self.Plots.Audio.Name = 'Audio'
    Self.Plots.C.Name = 'Harmonic Center'
    Self.Plots.V.Name = 'Variance'
    Self.Plots.H.Name = 'Harmoniousness'
    Self.Plots.CH.Name = 'Coharmoniousness'
    Self.Plots.HCH.Name = 'Both H and CoH'
    Self.Plots.HxCH.Name = 'H times CoH'
    # -----------------------

    print('building main window...')
    MainWindow = buildMainWindow(Self)
    MainWindow.show()
    print('running.')
    return App.exec_()


if __name__ == '__main__':
    sys.exit(Application())
