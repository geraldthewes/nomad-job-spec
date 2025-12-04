job "<job-name>" {
  datacenters = ["cluster"]

  # force redeploy
  meta {
    redeploy_uuid = "${uuidv4()}"
  }

  # hard code for now
  vault {
    policies = ["transcription-policy"]
  }

  group "service" {

    # Target nodes with GPU capability
    # constraint {
    #  attribute = "${meta.gpu-capable}"
    #  value     = "true"
    #}


    # Network mode host or bridged
    network {
      mode = "host"
      port "http" {}  # Dynamic port allocation; no 'static' or 'to'	 
    }

    # task
    task "server" {
      driver = "docker"

      resources {
        cpu    = 4000
        memory = 8192
      }

      config {
        image = "<docker-image>"
	network_mode = "host"  # Align Docker with Nomad's host mode
      }
      
      # Environment variables to set
      env {
      }

      # Vault template for secrets
      template {
        data = <<EOF
{{ with secret "<secret-path>" }}
SECRET1 = "{{ .Data.data.key1 }}"
SECRET2 = "{{ .Data.data.key2 }}"
{{ end }}
EOF
        destination = "secrets/<file>.env"
        env         = true
      }

      # Consul service information
      service {
        name = "<service-name>"
        port = "http"
        
        check {
          type     = "http"
          path     = "/health"
          interval = "10s"
          timeout  = "2s"
        }

       # used by fabio
       tags = [
        "<service-name>",
        "urlprefix-/transcribe strip=/transcribe"
      ]
      }
    }
    
  }
  

}
