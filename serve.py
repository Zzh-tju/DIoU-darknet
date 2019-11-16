#!/usr/bin/python
from twisted.web.server import Site, GzipEncoderFactory
from twisted.web.static import File
from twisted.web.resource import EncodingResourceWrapper
from twisted.internet import reactor

PORT = 8000
print("serving twisted reactor at port", PORT)

wrapped = EncodingResourceWrapper(Site(File(".")), [GzipEncoderFactory()])
site = Site(wrapped)
reactor.listenTCP(PORT, site)
reactor.run()

