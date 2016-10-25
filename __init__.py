# -*- coding: utf-8 -*-
"""
/***************************************************************************
 tower_predictor
                                 A QGIS plugin
 Predicts the optimal location of the tower
                             -------------------
        begin                : 2016-10-25
        copyright            : (C) 2016 by IIIT
        email                : kiran4399@gmail.com
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load tower_predictor class from file tower_predictor.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .tower_predictor import tower_predictor
    return tower_predictor(iface)
