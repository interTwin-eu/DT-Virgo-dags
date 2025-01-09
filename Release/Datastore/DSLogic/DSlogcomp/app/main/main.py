from flask import Blueprint

main_bp=Blueprint('main_bp',__name__)

from . import indexview ,errors,statview,dlogview,streamview,freezeview,cleanview,descview,dumpfview