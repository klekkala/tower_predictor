# -*- coding: utf-8 -*-
"""
/***************************************************************************
 tower_predictor
                                 A QGIS plugin
 Predicts the optimal location of the tower
                              -------------------
        begin                : 2016-10-25
        git sha              : $Format:%H$
        copyright            : (C) 2016 by IIIT
        email                : kiran4399@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from PyQt4.QtCore import QSettings, QTranslator, qVersion, QCoreApplication
from PyQt4.QtGui import QAction, QIcon
from qgis.core import QgsVectorLayer, QGis, QgsFeature, QgsPoint, QgsGeometry, QgsMapLayerRegistry

# Initialize Qt resources from file resources.py
import resources, optimize
# Import the code for the dialog
from tower_predictor_dialog import tower_predictorDialog
import os.path

#import optimize

class tower_predictor:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'tower_predictor_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

        # Create the dialog (after translation) and keep reference
        self.dlg = tower_predictorDialog()

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&tower_predictor')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'tower_predictor')
        self.toolbar.setObjectName(u'tower_predictor')

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('tower_predictor', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/tower_predictor/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'tower_predictor'),
            callback=self.run,
            parent=self.iface.mainWindow())


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&tower_predictor'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar


    def extract_features(self, layeriter):

        feat = []
        attr = []
        for feature in layeriter:
            geom = feature.geometry()
            print "Feature ID %d: " % feature.id()

            # show some information about the feature
            if geom.type() == QGis.Point:
                x = geom.asPoint()
                feat.append(x)
                attr.append(feature.attributes())

            elif geom.type() == QGis.Polygon:
                x = geom.asPolygon()
                feat.append(x)
                attr.append(feature.attributes())

        return feat, attr


    def draw_point(self, layer, point):
        pr = layer.dataProvider() 
        feat = QgsFeature()

        predict = QgsPoint(point[0], point[1])

        feat.setGeometry(QgsGeometry.fromPoint(predict))
        feat.setAttributes(['NULL', 50])

        pr.addFeatures([feat])
        layer.updateExtents()
        # add the layer to the canvas
        QgsMapLayerRegistry.instance().addMapLayers([layer])

    def load_layer(self):

        layers = self.iface.legendInterface().layers()

        #for layer in layers:

            #pop_feat, pop_attr = self.extract_features(layer.getFeatures())
            #print pop_feat

        celllayer = QgsVectorLayer("/home/kiran/Dropbox/cell.shp", "celltower", "ogr")
        cell_feat, cell_attr = self.extract_features(celllayer.getFeatures())
        cell_feat = [list(elem) for elem in cell_feat]
        attr_cell = []
        for attr in cell_attr:
            attr_cell.append(int(attr[1]))
        print cell_feat

        poplayer = QgsVectorLayer("/home/kiran/Dropbox/pop.shp", "population", "ogr")
        pop_feat, pop_attr = self.extract_features(poplayer.getFeatures())
        pop_feat = [list(elem) for elem in pop_feat]
        attr_pop = []
        for attr in pop_attr:
            attr_pop.append(int(attr[1]))
        print pop_feat

        elevlayer = QgsVectorLayer("/home/kiran/Dropbox/elev.shp", "elevation", "ogr")
        elev_feat, elev_attr = self.extract_features(elevlayer.getFeatures())
        elev_feat = [list(elem) for elem in elev_feat]
        attr_elev = []
        for attr in elev_attr:
            attr_elev.append(int(attr[1]))
        print elev_feat

        landlayer = QgsVectorLayer("/home/kiran/Dropbox/land.shp", "landcost", "ogr")
        land_feat, land_attr = self.extract_features(landlayer.getFeatures())
        attr_land = []
        for attr in land_attr:
            attr_land.append(int(attr[1]))

        cost_feat = []
        for each in land_feat:
            each = each[0]
            land_list = [list(elem) for elem in each]
            cost_feat.append(land_list)
        print cost_feat

        geomx = []
        geomy = []
        for geom in pop_feat:
            geomx.append(geom[0])
            geomy.append(geom[1])
        guess = [(sum(geomx)/len(geomx)), (sum(geomy)/len(geomy))]

        ret = optimize.minimize(guess, cell_feat, attr_cell, pop_feat, attr_pop, elev_feat, attr_elev, cost_feat, attr_land)
        #ret = optimize.testfun(guess)

        ##Draw the optimal location of cell tower
        newlayer = QgsVectorLayer("/home/kiran/Dropbox/new.shp", "optimal_tower", "ogr")
        self.draw_point(newlayer, ret)
        print "optimal location is"
        print ret


    def run(self):
        """Run method that performs all the real work"""
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            self.load_layer()
            pass
